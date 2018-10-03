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

class LiverData(torch_data.Dataset):
    def __init__(self, augment, paths=[]):
        self.augment = augment
        
        self.patient_paths = ['Data/3Dircadb/', 'Data/LiTS/Training/']
        if paths:
            self.patient_paths = paths
        
        self.num_datas = []
        for patient_path in self.patient_paths:
            self.num_datas.append(sum(-i.find('.') for i in os.listdir(patient_path)))
        self.num_data = sum(self.num_datas)
            
    def __len__(self):
        if self.augment:
            return self.num_data * 2
        else:
            return self.num_data
    
    def __getitem__(self, index):
        augment_voxel = False
        if self.augment and index >= self.num_data:
            augment_voxel = True
            index -= self.num_data
        
        patient_index = 0
        while index >= self.num_datas[patient_index]:
            index -= self.num_datas[patient_index]
            patient_index += 1
        
        patient_path = self.patient_paths[patient_index]
        
        patient = patient_path + 'patient{}/'.format(str(index+1))

        image_path = patient + 'image/'
        label_path = patient + 'label/'

        image_voxel, label_voxel = self.get_voxels(image_path, label_path, augment_voxel)
        
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
    
    def get_voxels(self, image_path, label_path, augment_voxel):
        scale, angle = self.gen_augment_values()
        
        image_voxel = self.get_voxel(image_path, scale, angle, augment_voxel)
        label_voxel = self.get_voxel(label_path, scale, angle, augment_voxel, label=True)
        return image_voxel, label_voxel
        
    def get_voxel(self, image_path, scale=1, angle=0, augment_voxel=False, label=False):
        imageIndex = 0
        voxel = list()
        while True:
            path = image_path + '{}.png'.format(str(imageIndex))
            if os.path.isfile(path):
                image = Image.open(path)
                image = image.resize((256, 256), Image.ANTIALIAS)
                if augment_voxel:
                    image = self.augmentation(image, scale, angle)
                    if not label:
                        image = T(image)
                image = ToTensor(image)
                voxel.append(image)
                imageIndex += 1
                continue
            else:
                break
        
        voxel = torch.stack(voxel)
        voxel = voxel.permute(1, 0, 2, 3)
        
        return voxel