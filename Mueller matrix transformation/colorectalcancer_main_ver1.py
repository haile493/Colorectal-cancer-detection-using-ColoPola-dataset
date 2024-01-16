# Calculating Mueller matrix transformation from 36 polarimetric images

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import csv
import cv2

# matplotlib.use('TkAgg')


def return_list_folder(path):
    listFolderName = os.listdir(path)
    listFolderPath = [os.path.join(path, x) for x in listFolderName]
    return listFolderPath, listFolderName


def save_image(parent_path, image_list, name_list):
    try:
        if not (os.path.exists(parent_path)):
            os.makedirs(parent_path)
        for index, name in enumerate(name_list):
            image_path = os.path.join(parent_path, name + ".png")
            cv2.imwrite(image_path, image_list[index])
    except:
        return False


def save4by4mueller(img_path, folder_name, save_path):
    varname = ['HH', 'HL', 'HM', 'HP', 'HR', 'HV',
               'LH', 'LL', 'LM', 'LP', 'LR', 'LV',
               'MH', 'ML', 'MM', 'MP', 'MR', 'MV',
               'PH', 'PL', 'PM', 'PP', 'PR', 'PV',
               'RH', 'RL', 'RM', 'RP', 'RR', 'RV',
               'VH', 'VL', 'VM', 'VP', 'VR', 'VV']
    images = [cv2.imread(file) for file in glob.glob(img_path)]

    # clear blue and green channels, keep red channel
    images_red_channel = []
    for img in images:
        # get red channel
        images_red_channel.append(img[..., 2])

    # combine polarization status and image
    m = dict(zip(varname, images_red_channel))
    # Apply mueller matrix
    m11 = m['HH'] + m['HV'] + m['VH'] + m['VV']
    m12 = m['HH'] + m['HV'] - m['VH'] - m['VV']
    m13 = m['PH'] + m['PV'] - m['MH'] - m['MV']
    m14 = m['RH'] + m['RV'] - m['LH'] - m['LV']
    m21 = m['HH'] - m['HV'] + m['VH'] - m['VV']
    m22 = m['HH'] - m['HV'] - m['VH'] + m['VV']
    m23 = m['PH'] - m['PV'] - m['MH'] + m['MV']
    m24 = m['RH'] - m['RV'] - m['LH'] + m['LV']
    m31 = m['HP'] - m['HM'] + m['VP'] - m['VM']
    m32 = m['HP'] - m['HM'] - m['VP'] + m['VM']
    m33 = m['PP'] - m['PM'] - m['MP'] + m['MM']
    m34 = m['RP'] - m['RM'] - m['LP'] + m['LM']
    m41 = m['HR'] - m['HL'] + m['VR'] - m['VL']
    m42 = m['HR'] - m['HL'] - m['VR'] + m['VL']
    m43 = m['PR'] - m['PL'] - m['MR'] + m['ML']
    m44 = m['RR'] - m['RL'] - m['LR'] + m['LL']

    # if m11 is 0, set average value of m11, if not set m11 value
    m11 = np.where(m11 == 0, np.average(m11), m11)

    mueller = [m11, m12, m13, m14,
               m21, m22, m23, m24,
               m31, m32, m33, m34,
               m41, m42, m43, m44]

    average = [1]
    for x in range(16):
        mueller[x] = mueller[x] / mueller[0]        
        # actually, this step is not necessary
        if x > 0:
            min_value = np.min(mueller[x])
            max_value = np.max(mueller[x])
            mueller[x] = 255 * (mueller[x] - min_value) / \
                (max_value - min_value)
            height_mueller, width_mueller = mueller[x].shape
            total_intensity = mueller[x][mueller[x] > 0].sum()
            average.append(
                round(total_intensity/(height_mueller*width_mueller*255), 4))                
                        
        # print(x, np.unique(mueller[x]))
    average = np.array(average).reshape(4, 4)
    np.savetxt(save_path + folder_name + '_4by4.csv', average, delimiter=",")

    # Concat each mueller to 4x4
    mueller[0] *= 255
    m1 = np.concatenate(
        (mueller[0], mueller[1], mueller[2], mueller[3]), axis=1)
    m2 = np.concatenate(
        (mueller[4], mueller[5], mueller[6], mueller[7]), axis=1)
    m3 = np.concatenate(
        (mueller[8], mueller[9], mueller[10], mueller[11]), axis=1)
    m4 = np.concatenate(
        (mueller[12], mueller[13], mueller[14], mueller[15]), axis=1)
    full_matrix = np.concatenate((m1, m2, m3, m4), axis=0)

    # Create figure and save in save_path folder
    fig, ax = plt.subplots(nrows=1, ncols=1)
    colorbar_range = [0, 0.25, 0.5, 0.75, 1]
    cbar = fig.colorbar(ax.imshow(full_matrix / 255., cmap='jet'),
                        fraction=0.046, pad=0.04, ticks=colorbar_range)
    cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])

    plt.axis('off')
    fig.savefig(save_path + folder_name + "_colorbar.png")
    fig.close()
    # plt.show()


if __name__ == '__main__':
    """
    Folder_path: Source path (with 36 images in each folder)
    Save_path: Destination path (to store output)
    TODO: folder_list = ['R3081V1'] nếu chỉ chạy 1 mẫu
    """

    # Benign, Normal, Stage2_cancer, Stage2_cancer
    sample_types = ['Cancer', 'Normal']
    save_path = "../results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for s in sample_types:
        folder_path = "../data/" + s + "/"
        folder_list = os.listdir(folder_path)
        # TODO with specific folder: folder_list = ['R3081V1']
        for folder in folder_list:
            folder_subpath = folder_path + folder + "/*.tif"
            print(folder_subpath)
            save4by4mueller(folder_subpath, folder, save_path)
            print("Success with " + folder)
        print("DONE for " + s)

