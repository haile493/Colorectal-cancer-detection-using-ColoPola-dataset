# Colorectal-cancer-detection-using-ColoPola-dataset

### Methods
We applied two models from scratch and two pretrained models (DenseNet121 and EfficientNetV2-m) to classify the colorectal cancer using the ColoPola dataset.

### ColoPola dataset
The dataset consists of 572 slices (specimens) with 20,592 images, 284 slices of which were designated as cancer samples and 288 as normal samples.
Repository: https://doi.org/10.5281/zenodo.10068031

![images of normal and malignant colorectal samples](https://github.com/haile493/Colorectal-cancer-detection-using-ColoPola-dataset/blob/main/images/polarimetric%20image.png)

Fig. 36 polarimetric images of (a) normal and (b) malignant colorectal samples in ColoPola dataset

### Requirements
- Pytorch 1.12.0 + cu116 or higher
- numpy 1.23.0 or higher
- torchmetrics 1.2.1
- scikit-learn 1.1.1
- albumentations 1.2.0 or higher
