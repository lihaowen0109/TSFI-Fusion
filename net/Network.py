import torch.nn.init as init
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import numpy as np
import numbers

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import pywt
import pywt.data


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class LocalIntegration(nn.Module):

    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """

    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer()
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):  # [1, 32, 128, 128]
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)  # [1, 64, 128, 128]
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [1, 32, 128, 128]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DDIM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, kernel_size=7):
        super(DDIM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.up = Conv1(in_channels, out_channels)

        # self.act = nn.SiLU()

    def forward(self, fre, spa):
        sub_input = fre - spa
        conv_sub = self.conv(sub_input)

        x_out_sam = self.spatial_attention(conv_sub)

        x_out_spa = (sub_input * (1 - x_out_sam)) + spa
        x_out_fre = (sub_input * x_out_sam) + fre

        x_out_concat = torch.cat((x_out_fre, x_out_spa), dim=1)

        x_out = self.channel_attention(x_out_concat)

        return x_out


class FDCM(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(FDCM, self).__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1)
        )
        self.norm = norm
        self.act = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape

        dim = 1

        y = torch.fft.rfft2(x, norm=self.norm)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)

        y = self.main_fft(y_f)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.act(y)
        y = y * x
        return y


class Frequency_Domain(nn.Module):
    def __init__(self, in_c, out_c, factor=4.0):
        super().__init__()
        self.out_c = out_c

        self.fam = FDCM(out_channel=out_c)

        self.conv1 = Conv1(in_channels=in_c * 2, out_channels=out_c)
        self.up1 = Conv1(in_channels=24, out_channels=64)

        self.cam = ChannelAttentionModule(in_channels=64)

        self.act = nn.Sigmoid()

    def forward(self, x):

        output_fam = self.fam(x)

        return output_fam


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, channel_out)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x2


class INN(nn.Module):
    def __init__(self, subnet_constructor=DenseBlock, clamp=2.0, harr=True, in_1=16, in_2=16):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 2
            self.split_len2 = in_2 * 2
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x):  # [1, 64, 128, 128]
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        t2 = self.f(x2)
        s2 = self.p(x2)
        y1 = self.e(s2) * x1 + t2
        s1, t1 = self.r(y1), self.y(y1)
        y2 = self.e(s1) * x2 + t1

        return torch.cat((y1, y2), 1)



class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for mod in self:
            x = mod(x)
        return x


class HOPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.horizontal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        vector_h = x.mean(dim=2, keepdim=True)
        vector_h = self.horizontal(vector_h)

        return vector_h


class VEPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.vertical = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        vector_v = x.mean(dim=3, keepdim=True)
        vector_v = self.vertical(vector_v)

        return vector_v


class MPE(nn.Module):
    def __init__(self, in_c, out_c, factor=4.0):
        super().__init__()
        self.out_c = out_c
        self.conv1 = Conv1(in_channels=in_c * 2, out_channels=out_c)
        self.up1 = Conv1(in_channels=24, out_channels=64)

        self.cam = ChannelAttentionModule(in_channels=64)

        self.act = nn.Sigmoid()
        # PWConv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        # Max Pooling
        self.maxpool = MaxPoolingChannel()

        # Avg Pooling
        self.avgpool = AvgPoolingChannel()

        # VE和HO Pooling
        self.vepool = VEPooling(in_channels=8, out_channels=8)
        self.hopool = HOPooling(in_channels=8, out_channels=8)

        # =============================================================================
        # WT Stream
        # =============================================================================
        self.wtconv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv2 = nn.Sequential(
            WTConv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )

        self.cam = ChannelAttentionModule(in_channels=64)

    def forward(self, x):
        identity = x
        x_pw = self.init_conv(x)

        x1, x2, x3, x4 = x_pw.chunk(4, dim=1)

        # branch1_Wtconv
        channel_1_1 = self.wtconv1(x1)
        channel_1_2 = self.wtconv2(channel_1_1)
        channel_1_3_out = self.wtconv3(channel_1_2)

        # # branch2_AvgPooling
        channel_2_avg_pool = self.avgpool(x2)
        desired_size2 = (x2.size(2), x2.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_avg_pool, size=desired_size2, mode='bilinear',
                                               align_corners=False)

        # branch3_HOPooling
        channel_3_ho_pool = self.hopool(x3)
        desired_size3 = (x2.size(2), x2.size(3))
        channel_3_ho_pool_out = F.interpolate(channel_3_ho_pool, size=desired_size3, mode='bilinear',
                                              align_corners=False)

        # branch4_VEPooling
        channel_4_ve_pool = self.vepool(x4)
        desired_size4 = (x2.size(2), x2.size(3))
        channel_4_ve_pool_out = F.interpolate(channel_4_ve_pool, size=desired_size4, mode='bilinear',
                                              align_corners=False)

        dualpool = channel_3_ho_pool_out + channel_4_ve_pool_out
        output_concat = torch.cat((channel_1_3_out, channel_2_avg_pool_out, dualpool), dim=1)
        output_concat = self.up1(output_concat)
        output_concat = self.cam(output_concat)
        output = identity + output_concat

        return output


class FFML(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(FFML, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1)
        )

        self.branch_fft = nn.Conv2d(in_channels=out_channel * 4, out_channels=out_channel * 2, kernel_size=1, stride=1)

        self.norm = norm
        self.act = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1

        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real

        y_f = torch.cat([y_real, y_imag], dim=dim)

        avg_out = self.avg_pool(y_f)
        max_out = self.max_pool(y_f)
        y_out = torch.cat([avg_out, max_out], dim=dim)

        y_main = self.main_fft(y_out)
        y_main = self.act(y_main)
        y_main = y_main * y_f

        y_out = self.branch_fft(y_out)
        y_main = y_main + y_out
        y_real, y_imag = torch.chunk(y_main, 2, dim=dim)
        y_main = torch.complex(y_real, y_imag)
        y_main = torch.fft.irfft2(y_main, s=(H, W), norm=self.norm)

        y_final = y_main + x
        return y_final


class DetailFeatureExtraction(nn.Module):
    def __init__(self, in_c, out_c, factor=4.0):
        super().__init__()
        self.out_c = out_c

        # 1.fourEmbed
        self.Four_Embed = MPE(in_c, out_c)

        # 2.frequency
        self.fre = Frequency_Domain(in_c=64, out_c=64)

        # 3.INN
        self.inn = INN()

        # 4.interaction
        self.interaction = DDIM(in_channels=in_c * 2, out_channels=out_c)

        self.Sobel_xy = Sobelxy(in_c)
        self.up = Conv1(in_c // 2, out_c)
        self.down1 = Conv1(in_channels=in_c * 2, out_channels=out_c)
        self.up1 = Conv1(in_channels=24, out_channels=64)
        self.act = nn.Sigmoid()

    def forward(self, x):
        output_embed = self.Four_Embed(x)
        output_fre = self.fre(output_embed)
        output_spa = self.inn(output_embed)
        output_inter = self.interaction(output_fre, output_spa)
        output_inter = self.down1(output_inter)
        output_sobel = self.Sobel_xy(x)
        output = output_inter + output_sobel

        return output


# =============================================================================
# Layer Norm
# =============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):  # (b,c,h,w)
        h, w = x.shape[-2:]
        """
        to_3d后：(b,h*w,c)
        body后：(b,h*w,c)
        to_4d后：(b,c,h,w)
        """
        return to_4d(self.body(to_3d(x)), h, w)


# =============================================================================
# Gated-Dconv Feed-Forward Network (GDFN)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# =============================================================================
# Multi-DConv Head Transposed Self-Attention (MDTA)
# =============================================================================

class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # LN->MDTA->残差连接
        x = x + self.ffn(self.norm2(x))  # LN->GDFN->残差连接
        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class CIIM(nn.Module):
    def __init__(self, dim):
        super(CIIM, self).__init__()
        self.conv_ir = nn.Conv2d(dim, dim * 2, 1, 1, bias=False)
        self.conv_vi = nn.Conv2d(dim, dim * 2, 1, 1, bias=False)
        self.conv_fusion = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.spatial_select = nn.Conv2d(dim * 6, 2, 1)

        self.sigmoid = nn.Sigmoid()

        self.DWConv = nn.Conv2d(in_channels=dim * 6, out_channels=dim, kernel_size=3, groups=dim, padding=1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.in_c = dim
        self.attention = eca_layer(channel=dim * 2, k_size=1)

    def forward(self, low, high):

        low_fre = self.conv_ir(low)
        high_fre = self.conv_vi(high)

        fuse = torch.cat([low_fre, high_fre], dim=1)
        fuse = self.conv_fusion(fuse)
        low_avg = low_fre.mean([2, 3], keepdim=True).expand(-1, -1, low_fre.shape[2], low_fre.shape[3])
        high_avg = high_fre.mean([2, 3], keepdim=True).expand(-1, -1, high_fre.shape[2], high_fre.shape[3])
        fuse = torch.cat([fuse, low_avg, high_avg], 1)

        h_map = self.DWConv(fuse)
        h_map = self.gelu(h_map)
        h_map = self.sigmoid(h_map)

        l_map = self.DWConv(fuse)
        l_map = self.relu(l_map)
        l_map = self.sigmoid(l_map)

        output = torch.cat((h_map * low, l_map * high), dim=1)

        output = output.permute(0, 2, 3, 1).contiguous()
        output = channel_shuffle(output, groups=self.in_c // 2)
        output = output.permute(0, 3, 1, 2).contiguous()
        output = self.attention(output)

        return output


# =============================================================================
# Overlapped image patch embedding with 3x3 Conv
# =============================================================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.baseFeature = BaseFeatureExtraction(dim=dim)
        self.detailFeature = DetailFeatureExtraction(in_c=64, out_c=64)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)

        return base_feature, detail_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()

        self.reduce_channel1 = CIIM(dim=64)
        self.reduce_channel2 = nn.Conv2d(in_channels=int(dim * 2), out_channels=int(dim), kernel_size=1, bias=bias)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim), out_channels=int(dim) // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(dim) // 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature, fuse='MIFA_block'):
        if fuse == 'MIFA_block':
            out_enc_level0 = self.reduce_channel1(base_feature, detail_feature)
            out_enc_level0 = self.reduce_channel2(out_enc_level0)
        else:
            out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
            out_enc_level0 = self.reduce_channel2(out_enc_level0)

        out_enc_level1 = self.encoder_level2(out_enc_level0)

        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()
