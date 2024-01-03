# Crop MM images from 1024x1280 (HxW) to 900x900 pixels

import os
import cv2
from pathlib import Path

fullsize_path = r"C:\CC_ViHiep\data_fullsize"
smallsize_path = r"C:\CC_ViHiep\data_small"
polarstates = ['HH', 'HL', 'HM', 'HP', 'HR', 'HV',
               'LH', 'LL', 'LM', 'LP', 'LR', 'LV',
               'MH', 'ML', 'MM', 'MP', 'MR', 'MV',
               'PH', 'PL', 'PM', 'PP', 'PR', 'PV',
               'RH', 'RL', 'RM', 'RP', 'RR', 'RV',
               'VH', 'VL', 'VM', 'VP', 'VR', 'VV']


def return_list_folder(path):
    listFolderName = os.listdir(path)
    listFolderPath = [os.path.join(path, x) for x in listFolderName]
    return listFolderPath, listFolderName


def crop_36_polarstates():
    # crop images from 1024x1280 to 900x900
    source = fullsize_path
    destination = smallsize_path
    folder_paths, folder_names = return_list_folder(source)    
    save_folder_paths = [os.path.join(destination, x) for x in folder_names]
    # print(save_folder_paths)
    for idx, fpath in enumerate(folder_paths):
        sample_paths, sample_names = return_list_folder(fpath)        
        for i, sample_path in enumerate(sample_paths):
            print(sample_paths[i])
            for fn in polarstates:                
                img_full = cv2.imread(os.path.join(sample_path, fn + '.tif'))                
                if img_full is None:
                    print(os.path.join(sample_path, fn + '.tif'))
                    continue
                height, width, _ = img_full.shape
                center_location = [height//2, width//2]
                img_crop = img_full[(center_location[0] - 450):(center_location[0] + 450),
                           (center_location[1] - 450):(center_location[1] + 450)]
                save_path = os.path.join(save_folder_paths[idx], sample_names[i])                
                if not (os.path.exists(save_path)):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, fn + '.png'), img_crop)

    print("Done!")


