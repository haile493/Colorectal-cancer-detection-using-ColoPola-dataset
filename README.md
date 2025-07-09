# Colorectal-cancer-detection-using-ColoPola-dataset

### Methods
We trained and tested three models from scratch (CNN, CNN_2 and EfficientFormerV2) and two pretrained models (DenseNet121 and EfficientNetV2-m) to classify the colorectal cancer using the ColoPola dataset.

### ColoPola dataset
The dataset consists of 572 slices (specimens) with 20,592 images. There are 284 cancer samples and 288 normal samples.
This dataset can download from [Zenodo repository](https://doi.org/10.5281/zenodo.10068018). The lists of samples for training (train+val) set and testing set are provided in this repository.

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

### Quick Start
1. Download the ColoPola dataset that contains all samples with train.txt and test.txt files
2. Split the training set (list of samples in train.txt) into train and validation sets at any desired ratio. Keep the testing set (test.txt) for evaluating the trained model(s) as unseen data.
3. Install packages in requirements_short.txt
4. Modify '''cc_model.py''' to select one of three models from scratch and one of two pretrained models. Then make sure the paths for train.txt and val.txt
5. Run main.py with pretrained = True or False. Then set the paths for val.txt and test.txt to evaluate the trained models.

### Notes
- In this study, the input shape is (224, 224, 36) with 36 channels that are 36 polarized images for each slice. Please read cc_dataset.py to know how to make the input data.
- The architectures of five models are in build_model_2.py for CNN and CNN_2, efficientformer_v2.py for EfficientFormerV2, and pretrained_models.py for DenseNet121 and EfficientNetV2.


