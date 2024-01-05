# Le Thanh Hai
# June 07, 2023
# Classify 1 colorectal cancer class and 1 normal class (ColoPola dataset)
# Use all of 36 polarized images for classification

from crop_img import crop_36_polarstates
from cc_model import train_model, test_trained_model


if __name__ == '__main__':
    # crop full size images to 900x900 images
    # crop_36_polarstates()
    
    train_model(pretrained=True)
    # train_model(pretrained=False)
    
    # test_trained_model(model=None, fname='valid.txt')
    # test_trained_model(model=None, fname='test.txt')


