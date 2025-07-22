## **TSFI-Fusion: A dual-branch decoupled infrared and visible image fusion network based on Transformer and spatial-frequency interaction**



## Abstract

> Infrared and visible image fusion (IVIF) aims to generate high-quality images by combining detailed textures from visible images with the target-highlight capabilities of infrared images. However, many existing methods struggle to capture both shared and unique features of each modality. They often focus only on spatial domain fusion, like pixel averaging, and overlook valuable frequency domain information—making it hard to retain fine details. To overcome these limitations, we propose TSFI-Fusion, a dual branch network that combines Transformer-based global understanding with spatial-frequency detail enhancement. The two branches include a Transformer-based semantic construction branch for capturing global features and a detail enhancement branch utilizing an invertible neural network (INN) and a frequency domain compensation module (FDCM) to integrate spatial and frequency information. We also design a dual-domain interaction module (DDIM) to improve feature correlation across domains and a collaborative information integration module (CIIM) to effectively merge features from both branches. Additionally, we introduce a focal frequency loss to guide the model in learning important frequency information. Experimental results demonstrate that TSFI-Fusion outperforms existing methods across multiple datasets and metrics on the IVIF task. In downstream applications such as object detection, it effectively enhances performance. Furthermore, extended experiments on the MIF task reveal the robust generalization ability of the proposed mechanism across diverse fusion scenarios.

## Network Architecture

The structure of **TSFI-Fusion** is illustrated in the figure below:


![image](https://github.com/leesir2001/TSFI-Fusion/blob/main/img/The%20network%20architecture.png)


## Dataset
To begin, please first acquire the datasets. This project uses four publicly available infrared-visible image fusion datasets:
- **MSRS**：https://github.com/Linfeng-Tang/MSRS
- **TNO**：https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029
- **M3FD**： https://github.com/dlut-dimt/TarDAL
- **Harvard medical  dataset**：https://www.med.harvard.edu/AANLIB/home.html

Please refer to the official sources of each dataset for download and usage instructions.

## To Train

Please prepare your dataset and process it by running the dataprocessing.py, then modify the path to the dataset in the train file and run the following file to start training.

```python
python train.py
```

## To Test

```python
python test.py
```


## Related work

```
@inproceedings{zhao2023cddfuse,
  title={Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion},
  author={Zhao, Zixiang and Bai, Haowen and Zhang, Jiangshe and Zhang, Yulun and Xu, Shuang and Lin, Zudi and Timofte, Radu and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5906--5916},
  year={2023}
}
```


## Acknowledgements

The codes are based on [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse), and [CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT). Please also follow their licenses. Thanks for their awesome work.
