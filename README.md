# Colorectal-cancer-detection-using-ColoPola-dataset

### Methods
We trained and tested two models from scratch and two pretrained models (DenseNet121 and EfficientNetV2-m) to classify the colorectal cancer using the ColoPola dataset.

### ColoPola dataset
The dataset consists of 572 slices (specimens) with 20,592 images. There are 284 cancer samples and 288 normal samples.
This dataset can download from [Zenodo repository](https://doi.org/10.5281/zenodo.10068031). The lists of samples for training (train+val) set and testing set are provided in this repository.

![normal](https://github.com/haile493/Colorectal-cancer-detection-using-ColoPola-dataset/blob/main/images/normal.png)
<p align="center">Fig. 1 36 polarimetric images of normal sample</p>

![cancer](https://github.com/haile493/Colorectal-cancer-detection-using-ColoPola-dataset/blob/main/images/cancer.png)
<p align="center">Fig. 2 36 polarimetric images of malignant colorectal sample</p>

### Requirements
- Python >= 3.9
- Pytorch >= 1.12.0 + cu116
- numpy >= 1.23.0
- torchmetrics >= 1.2.1
- scikit-learn >= 1.1.1
- albumentations >= 1.2.0
