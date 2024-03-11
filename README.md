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
### 2.1 NYU:
-Change delta as 0.4 and split (in Line 9) based on your GPU memory (split=1 requires huge memory about ~40 GB) in [mmseg/model/losses/ap_loss.py](https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/ap_loss.py#L9-10)

        
-Run the following command to start training.

```shell
python tools/train.py configs/APLoss/base_320_fullData.py --options model.pretrained=preTrain/swin_base_patch4_window12_384_22k.pth model.backbone.use_checkpoint=True --work-dir your_folder
```

### 2.2 BSDS:

#### For Only Ranking:
-Change delta as 0.1 and split (in Line 9) based on your GPU memory (split=1 requires huge memory about ~40 GB) in [mmseg/model/losses/ap_loss.py](https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/ap_loss.py#L9-10)

        
-Run the following command to start training.

```shell
python tools/train.py configs/APLoss/base_320_fullData_bsds.py --options model.pretrained=preTrain/swin_base_patch4_window12_384_22k.pth model.backbone.use_checkpoint=True --work-dir your_folder
```

#### For Ranking & Sorting:
-Change delta as 0.1 and split (in Line 9) based on your GPU memory (split=1 requires huge memory about ~40 GB) in [mmseg/model/losses/rank_loss.py]([https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/ap_loss.p](https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/rank_loss.py)y#L9-10) and [mmseg/model/losses/sort_loss.py]([https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/ap_loss.p](https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/models/losses/sort_loss.py)y#L9-10)

- Comment the line 12 in [mmseg/datasets/pipelines/bsds_get_gtfiles.py](https://github.com/Bedrettin-Cetinkaya/RankED/blob/main/mmseg/datasets/pipelines/bsds_get_gtfiles.py#L12-13)
 
-Run the following command to start training.

```shell
python tools/train.py configs/RSLoss/base_320_fullData_bsds.py --options model.pretrained=preTrain/swin_base_patch4_window12_384_22k.pth model.backbone.use_checkpoint=True --work-dir your_folder
```
## 3. Inference

-Run the following command to start inference. 
python tools/test.py --config configs/APLoss/base_320_fullData_bsds.py --checkpoint your_folder/xxx.pth --tmpdir your_save_result_dir

## 4. Acknowledgements
Thanks to the previous open-sourced repo:

[EDTER](https://github.com/MengyangPu/EDTER)

[Swin-Tranformer](https://github.com/microsoft/Swin-Transformer)

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)

## 5. Reference
```bibtex
@InProceedings{cetinkaya2024ranked,
  title={RankED: Addressing Imbalance and Uncertainty in Edge Detection Using Ranking-based Losses}, 
  author={Bedrettin Cetinkaya and Sinan Kalkan and Emre Akbas},
  year={2024},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  url={https://ranked-cvpr24.github.io/}
}
```

