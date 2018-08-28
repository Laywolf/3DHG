import os
import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from torchvision import transforms

from Reader.util import rand

import math
import random

T = transforms.Compose([
    transforms.ColorJitter(0.3, 0, 0, 0)
])
ToTensor = transforms.ToTensor()

class Liver3LayerSH(torch_data.Dataset):
    def __init__(self, augment):
        self.augment = augment
        
        self.patient_path = 'Data/LiTS/Training/'
        #self.patient_path = 'Data/3Dircadb/'
        
        self.num_data = 0
        self.num_data_list = []
        dir_list = os.listdir(self.patient_path)
        dir_list.sort(key=lambda f: int(f[7:]))
        for patient in dir_list:
            path = self.patient_path + patient + '/image'
            if path.find('.') < 0:
                num = len(os.listdir(path))
                self.num_data += math.ceil(num / 3)
                self.num_data_list.append(num)
            
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        #augmentation
        augment_voxel = False
        if random.random() > 0.6:
            augment_voxel = True
        
        #patient index
        patient_index = -1
        while index > -1:
            patient_index += 1
            index -= math.ceil(self.num_data_list[patient_index] / 3)
        
        #crop index
        crop_index = index + math.ceil(self.num_data_list[patient_index] / 3)
        crop_index = int(crop_index * 3)
        if crop_index > self.num_data_list[patient_index]-3:
            crop_index = self.num_data_list[patient_index]-3
        
        patient = self.patient_path + 'patient{}/'.format(str(patient_index + 1))

        image_path = patient + 'image/'
        label_path = patient + 'label/'

        image_voxel, label_voxel = self.get_voxels(
            image_path, label_path,
            augment_voxel, crop_index
        )
        
        return image_voxel, label_voxel
        
    def __add__(self, item):
        pass
    
    def gen_augment_values(self):
        scale = 2 ** rand(0.25) * 1.25
        angle = rand(30)
        
        return scale, angle
        
    def augmentation(self, image, scale, angle):
        new_image = image
        resolution = image.width
        if angle is not 0:
            new_image = new_image.rotate(angle, resample=Image.BILINEAR)
            left = (new_image.width - resolution) / 2
            top = (new_image.height - resolution) / 2
            new_image = new_image.crop(
                box=(
                    left, top,
                    resolution,
                    resolution
                )
            )
            
            new_image.resize((resolution, resolution), Image.BILINEAR)
        return new_image
    
    def get_voxels(self, image_path, label_path, augment_voxel, crop_index):
        scale, angle = self.gen_augment_values()
        
        image_voxel = self.get_voxel(image_path, crop_index,
                                     scale, angle, augment_voxel)
        label_voxel = self.get_voxel(label_path, crop_index,
                                     scale, angle, augment_voxel, label=True)
        return image_voxel, label_voxel
        
    def get_voxel(self, image_path, crop_index,
                  scale=1, angle=0, augment_voxel=False, label=False):
        
        voxel = list()
        for index in range(crop_index, crop_index + 3):
            path = image_path + '{}.png'.format(str(index))
            image = Image.open(path)
            if augment_voxel:
                image = self.augmentation(image, scale, angle)
                if not label:
                    image = T(image)
            image = ToTensor(image)
            voxel.append(image)
        
        voxel = torch.stack(voxel)
        voxel = voxel.permute(1, 0, 2, 3)
        
        return voxel