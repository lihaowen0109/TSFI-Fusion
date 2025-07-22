from net.Network import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import cv2
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r"./models/IVIF_MSRS.pth"
# for dataset_name in ["TNO","RoadScene50","MSRS","M3FD","FMB"]:
# for dataset_name in ["TNO","MSRS","M3FD","FMB"]:
for dataset_name in ["TNO"]:
    print("\n" * 2 + "=" * 80)
    model_name = "TSFI-Fusion    "
    print("The test result of " + dataset_name + ' :')

    path = './test_img/'
    test_folder = os.path.join(path, dataset_name)
    test_out_folder = os.path.join(r'./test_result/TNO1', dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)

    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(64, 64)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            # 灰色
            # data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            # 彩色
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][
                           np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            # feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            # feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(data_VIS, feature_V_B + feature_I_B, feature_V_D + feature_I_D, fuse='MIFA_block')
            # data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D, fuse='MIFA_block')

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            # 灰色
            # img_save(fi, img_name.split(sep='.')[0], test_out_folder)
            # 彩色
            fi = fi.astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)

