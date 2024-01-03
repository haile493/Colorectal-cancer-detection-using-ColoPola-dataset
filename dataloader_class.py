# import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import List
from torch.utils.data import DataLoader

from cc_dataset import DatasetGenerator


def get_loaders(
        image_dir: Path,
        train_path: Path,
        valid_path: Path,
        batch_size: int = 32,
        num_workers: int = 0,  # 4
        train_transforms_fn=None,
        valid_transforms_fn=None
) -> dict:

    # Creates our train dataset
    train_dataset = DatasetGenerator(pathImageDirectory=image_dir,
                                     pathDatasetFile=train_path,
                                     transform=train_transforms_fn
                                     )
    valid_dataset = DatasetGenerator(pathImageDirectory=image_dir,
                                     pathDatasetFile=valid_path,
                                     transform=valid_transforms_fn
                                     )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=False
                              )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False
                              )

    # And expect to get an OrderedDict of loaders
    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def get_test_loaders(
        image_dir: Path,
        test_path: Path,
        batch_size: int = 1,
        num_workers: int = 0,  # 4
        test_transforms_fn=None,
) -> dict:

    # Creates our test dataset
    test_dataset = DatasetGenerator(pathImageDirectory=image_dir,
                                    pathDatasetFile=test_path,
                                    transform=test_transforms_fn
                                    )

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers
                             )

    loader = OrderedDict()
    loader["test"] = test_loader

    return loader

