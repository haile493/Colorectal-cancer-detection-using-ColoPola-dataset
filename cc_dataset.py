import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.listimagepaths = []
        self.listlabels_onehot = []
        self.transform = transform
        self.fnames = []
        self.polarstates = ['HH', 'HL', 'HM', 'HP', 'HR', 'HV',
                            'LH', 'LL', 'LM', 'LP', 'LR', 'LV',
                            'MH', 'ML', 'MM', 'MP', 'MR', 'MV',
                            'PH', 'PL', 'PM', 'PP', 'PR', 'PV',
                            'RH', 'RL', 'RM', 'RP', 'RR', 'RV',
                            'VH', 'VL', 'VM', 'VP', 'VR', 'VV']
        self.polar_imgs = dict.fromkeys(self.polarstates)

        # ---- Open file, get image paths and labels
        with open(pathDatasetFile, 'r') as f:
            for line in f:
                # read a line with file path and labels (Muller matrix parameters)
                items = line.split()                
                # get a filenames from 0, others are labels
                polar_paths = [items[0] + '\\' + polar + '.png' for polar in self.polarstates]
                fnames = [os.path.join(pathImageDirectory, polar_path) for polar_path in polar_paths]
                
                if items[0].find('normal') != -1:                    
                    self.listlabels_onehot.append([0])
                elif items[0].find('cancer') != -1:                    
                    self.listlabels_onehot.append([1])
                
                self.listimagepaths.append(fnames)                
                self.fnames.append(items[0])

        f.close()        

    def __len__(self):
        return len(self.listimagepaths)

    def __getitem__(self, index):
        image_path = self.listimagepaths[index]
        for idx in range(len(image_path)):
            # print(image_path[idx])
            if not os.path.isfile(image_path[idx]):
                print(f'File does not exist: {image_path[idx]}')                
            else:
                img = cv2.imread(image_path[idx])
                polar_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.polar_imgs[self.polarstates[idx]] = polar_img                

        # ---- Get label --------
        label_onehot = torch.FloatTensor(self.listlabels_onehot[index])        

        if self.transform is not None:
            # After transforming, augmented is torch tensor
            augmented = self.transform(image=self.polar_imgs['HH'], image1=self.polar_imgs['HL'],
                                       image2=self.polar_imgs['HM'], image3=self.polar_imgs['HP'],
                                       image4=self.polar_imgs['HR'], image5=self.polar_imgs['HV'],
                                       image6=self.polar_imgs['LH'], image7=self.polar_imgs['LL'],
                                       image8=self.polar_imgs['LM'], image9=self.polar_imgs['LP'],
                                       image10=self.polar_imgs['LR'], image11=self.polar_imgs['LV'],
                                       image12=self.polar_imgs['MH'], image13=self.polar_imgs['ML'],
                                       image14=self.polar_imgs['MM'], image15=self.polar_imgs['MP'],
                                       image16=self.polar_imgs['MR'], image17=self.polar_imgs['MV'],
                                       image18=self.polar_imgs['PH'], image19=self.polar_imgs['PL'],
                                       image20=self.polar_imgs['PM'], image21=self.polar_imgs['PP'],
                                       image22=self.polar_imgs['PR'], image23=self.polar_imgs['PV'],
                                       image24=self.polar_imgs['RH'], image25=self.polar_imgs['RL'],
                                       image26=self.polar_imgs['RM'], image27=self.polar_imgs['RP'],
                                       image28=self.polar_imgs['RR'], image29=self.polar_imgs['RV'],
                                       image30=self.polar_imgs['VH'], image31=self.polar_imgs['VL'],
                                       image32=self.polar_imgs['VM'], image33=self.polar_imgs['VP'],
                                       image34=self.polar_imgs['VR'], image35=self.polar_imgs['VV'],
                                       )
            img_aug = augmented['image'][0, :, :]            
            img_aug = img_aug[None, ...]            
            for idx in range(1, len(image_path)):
                img_name = 'image' + str(idx)                
                aug = augmented[img_name][0, :, :]  # red channel, torch tensor                
                img_aug = torch.cat([img_aug, aug[None, ...]], dim=0)                
            # visualize_augmentations(np.stack((flair_aug, t1_aug, t1ce_aug, t2_aug), axis=0))
            
            result = {'images': img_aug, 'targets': label_onehot, 'filename': self.fnames[index]}

        return result


class Mean_Std_DatasetGenerator(Dataset):
    # for calculating mean and std of dataset
    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
        self.listimagepaths = []
        self.transform = transform
        self.fnames = []
        self.polarstates = ['HH', 'HL', 'HM', 'HP', 'HR', 'HV',
                            'LH', 'LL', 'LM', 'LP', 'LR', 'LV',
                            'MH', 'ML', 'MM', 'MP', 'MR', 'MV',
                            'PH', 'PL', 'PM', 'PP', 'PR', 'PV',
                            'RH', 'RL', 'RM', 'RP', 'RR', 'RV',
                            'VH', 'VL', 'VM', 'VP', 'VR', 'VV']
        self.polar_imgs = dict.fromkeys(self.polarstates)

        # ---- Open file, get image paths and labels
        with open(pathDatasetFile, 'r') as f:
            for line in f:
                # read a line with file path and one hot label (Muller matrix parameters)
                items = line.split()                
                # get a filenames from 0, others are labels
                polar_paths = [items[0] + '\\' + polar + '.png' for polar in self.polarstates]
                fname = [os.path.join(pathImageDirectory, polar_path) for polar_path in polar_paths]
                
                self.listimagepaths.append(fname)
                self.fnames.append(items[0])

        f.close()        

    def __len__(self):
        return len(self.listimagepaths)

    def __getitem__(self, index):
        image_path = self.listimagepaths[index]
        for idx in range(len(image_path)):            
            if not os.path.isfile(image_path[idx]):
                print(f'File does not exist: {image_path[idx]}')                
            else:
                img = cv2.imread(image_path[idx])
                polar_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.polar_imgs[self.polarstates[idx]] = polar_img

        if self.transform is not None:
            augmented = self.transform(image=self.polar_imgs['HH'], image1=self.polar_imgs['HL'],
                                       image2=self.polar_imgs['HM'], image3=self.polar_imgs['HP'],
                                       image4=self.polar_imgs['HR'], image5=self.polar_imgs['HV'],
                                       image6=self.polar_imgs['LH'], image7=self.polar_imgs['LL'],
                                       image8=self.polar_imgs['LM'], image9=self.polar_imgs['LP'],
                                       image10=self.polar_imgs['LR'], image11=self.polar_imgs['LV'],
                                       image12=self.polar_imgs['MH'], image13=self.polar_imgs['ML'],
                                       image14=self.polar_imgs['MM'], image15=self.polar_imgs['MP'],
                                       image16=self.polar_imgs['MR'], image17=self.polar_imgs['MV'],
                                       image18=self.polar_imgs['PH'], image19=self.polar_imgs['PL'],
                                       image20=self.polar_imgs['PM'], image21=self.polar_imgs['PP'],
                                       image22=self.polar_imgs['PR'], image23=self.polar_imgs['PV'],
                                       image24=self.polar_imgs['RH'], image25=self.polar_imgs['RL'],
                                       image26=self.polar_imgs['RM'], image27=self.polar_imgs['RP'],
                                       image28=self.polar_imgs['RR'], image29=self.polar_imgs['RV'],
                                       image30=self.polar_imgs['VH'], image31=self.polar_imgs['VL'],
                                       image32=self.polar_imgs['VM'], image33=self.polar_imgs['VP'],
                                       image34=self.polar_imgs['VR'], image35=self.polar_imgs['VV'],
                                       )

            img_aug = augmented['image'][0, :, :]  # red channel            
            img_aug = img_aug[None, ...]            
            for idx in range(1, len(image_path)):
                img_name = 'image' + str(idx)                
                aug = augmented[img_name][0, :, :]  # red channel, torch tensor                                
                img_aug = torch.cat([img_aug, aug[None, ...]], dim=0)
            
        return img_aug
