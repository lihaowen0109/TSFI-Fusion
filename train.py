# -*- coding: utf-8 -*-

from net.Network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc, FocalFrequencyLoss
import kornia

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# =============================================================================
# Configure our network
# =============================================================================

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'TSFI-Fusion'

# Set the hyper-parameters for training
num_epochs = 120  # total epoch
epoch_gap = 40  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

# Coefficients of the loss function
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

coeff_fre_loss_IF = 1.0
coeff_fre_loss_VF = 1.0

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(in_c=64, out_c=64)).to(device)


optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)


scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)


MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')
FreLoss = FocalFrequencyLoss()

trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# =============================================================================
# Train
# =============================================================================
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap:  # Phase I

            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)

            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D, fuse='concat')
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D, fuse='concat')

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))


            fre_loss_V = FreLoss(data_VIS, data_VIS_hat)
            fre_loss_I = FreLoss(data_IR, data_IR_hat)


            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + coeff_decomp * loss_decomp \
                   + coeff_tv * Gradient_loss + coeff_fre_loss_IF * fre_loss_I + coeff_fre_loss_VF * fre_loss_V
            loss.backward()

            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()

        else:  # Phase II

            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)

            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_I_B + feature_V_B, feature_I_D + feature_V_D, fuse='MIFA_block')

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()

            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()


        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] [Batch {i + 1}/{len(loader['train'])}] [loss: {loss.item()}] ETA: {time_left}")

    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(
        f"./models/model_{timestamp}_epoch_{epoch + 1}.pth"))


