# RankED: Addressing Imbalance and Uncertainty in Edge Detection Using Ranking-based Losses (CVPR 2024)

## 1. Install Environment, Datasets, Pretrained Model
### 1.1 Environment
Our project is based-on MMSegmentation. Please follow the official MMsegmentation [INSTALL.md](https://github.com/open-mmlab/mmsegmentation/blob/v0.11.0/docs/get_started.md#installation) with Python= 3.7, Pytorch= 1.12.1, CUDA=11.3 and CUDNN=8.3.2.

### 1.2 Datasets
#### NYUD:

Download the augmented NYUD data from [EDTER](https://github.com/MengyangPu/EDTER) Repository:

-Download [Train Data](https://drive.google.com/drive/folders/1lTfTIS-vlTtId-LGhEO2ZZjonA3SmLGJ)

-Download [Test Data](https://drive.google.com/drive/folders/1TQpKzCV4Ujkfs4V_vMasKAcvg3p4ByCN)

Put these files into data/NYUD/.

#### BSDS:

Download the augmented BSDS data from [here](https://drive.google.com/drive/folders/16W1yK8LpbJNin5C8_HLTt_w5pz25kLOs?usp=sharing)

Put these file into data/BSDS_RS/.

Extract these files via tar zxvf filename.tar.gz

### 1.3 Initial Weights

Download the pretrained model from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)

Put this file into preTrain/.

## 2. Train Model
