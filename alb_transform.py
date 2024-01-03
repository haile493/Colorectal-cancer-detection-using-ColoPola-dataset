import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import cv2

# mean and std of training set when training the models from scratch
mean0 = (0.2679, 0.2679, 0.2679)
std0 = (0.2679, 0.2679, 0.2679)


def pre_transforms(image_size=224):  # default=299 for xception, 224
    return [albu.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        # albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # albu.GridDistortion(p=0.3),
        # albu.HueSaturationValue(p=0.3),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.OneOf([
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                  rotate_limit=15,
                                  border_mode=cv2.BORDER_CONSTANT, value=0),
            albu.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   value=0),
            albu.NoOp()
        ]),

        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.5,
                                          contrast_limit=0.4),
            albu.RandomGamma(gamma_limit=(50, 150)),
            albu.NoOp()
        ]),

        albu.OneOf([
            albu.CLAHE(),
            albu.NoOp()
        ]),

        # albu.OneOf([
        #     albu.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
        #     albu.HueSaturationValue(),
        #     albu.NoOp()
        # ]),

        albu.OneOf([
            albu.Sharpen(),
            albu.Blur(blur_limit=3),
            albu.MotionBlur(blur_limit=3),
            albu.NoOp()
        ]),

        # albu.OneOf([
        #     albu.RandomFog(),
        #     albu.RandomSunFlare(src_radius=100),
        #     albu.RandomRain(),
        #     albu.RandomSnow(),
        #     albu.NoOp()
        # ]),
    ]

    return result


def resize_transforms(image_size=224):  # default=299 for xception, 224, 384 for efficientnet
    # BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(image_size, image_size, p=1)
    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(image_size, image_size, p=1)
    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big], p=1)
    ]

    return result


def post_transforms():
    return [albu.Normalize(mean=mean0, std=std0), ToTensorV2()]
    

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist],
        additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image',
                            'image4': 'image', 'image5': 'image', 'image6': 'image',
                            'image7': 'image', 'image8': 'image', 'image9': 'image',
                            'image10': 'image', 'image11': 'image', 'image12': 'image',
                            'image13': 'image', 'image14': 'image', 'image15': 'image',
                            'image16': 'image', 'image17': 'image', 'image18': 'image',
                            'image19': 'image', 'image20': 'image', 'image21': 'image',
                            'image22': 'image', 'image23': 'image', 'image24': 'image',
                            'image25': 'image', 'image26': 'image', 'image27': 'image',
                            'image28': 'image', 'image29': 'image', 'image30': 'image',
                            'image31': 'image', 'image32': 'image', 'image33': 'image',
                            'image34': 'image', 'image35': 'image', 'image36': 'image',
                            }
    )
    return result


def prepare_transform():
    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])

    valid_transforms = compose([pre_transforms(), post_transforms()])

    show_transforms = compose([resize_transforms(), hard_transforms()])

    return train_transforms, valid_transforms, show_transforms


def transform_mean_std(image_size=224):
    # We use A.Normalize() with mean = 0 and std = 1 to scale pixel values from [0, 255] to [0, 1]
    # and ToTensorV2() to convert numpy arrays into torch tensors.
    img_transform = albu.Compose([albu.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_NEAREST),
                                  albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                                  ToTensorV2()],
                                 additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image',
                                                     'image4': 'image', 'image5': 'image', 'image6': 'image',
                                                     'image7': 'image', 'image8': 'image', 'image9': 'image',
                                                     'image10': 'image', 'image11': 'image', 'image12': 'image',
                                                     'image13': 'image', 'image14': 'image', 'image15': 'image',
                                                     'image16': 'image', 'image17': 'image', 'image18': 'image',
                                                     'image19': 'image', 'image20': 'image', 'image21': 'image',
                                                     'image22': 'image', 'image23': 'image', 'image24': 'image',
                                                     'image25': 'image', 'image26': 'image', 'image27': 'image',
                                                     'image28': 'image', 'image29': 'image', 'image30': 'image',
                                                     'image31': 'image', 'image32': 'image', 'image33': 'image',
                                                     'image34': 'image', 'image35': 'image', 'image36': 'image',
                                                     }
                                 )

    return img_transform

