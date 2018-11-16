#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torchvision import transforms

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import nibabel as nib
import pydicom

from dotmap import DotMap

config = DotMap({
    "training": True,
    "batch": 32,
    "workers": 0,
    "epoch": 200,
    "in_scale": 256,
    "out_scale": 256
})

#Read NII file
def readNII(filePath):
    #LITS reader
    print('Path to the NIFTI file: {}'.format(filePath))
    
    #read nii
    image = nib.load(filePath)
    
    voxel = np.asarray(image.get_fdata())
    
    voxel = voxel.transpose()
    voxel = np.flip(voxel, 1)
    
    return voxel

def windowing(voxel):
    windowLow = -100
    windowHigh = 400
    
    voxel = np.clip(voxel, windowLow, windowHigh)
    voxel = (voxel - windowLow) / (windowHigh - windowLow) * 255
    
    return voxel

# network structure start -----------------------------------------------
# 1x1 convolution
def conv_one(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channel, out_channel, 1,
                  bias=False)
    )

# 3x3 convolution
def conv_thr(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm3d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channel, out_channel, 3,
                  padding=1, bias=False)
    )

# dilated convolution
def conv_dil(in_channel, out_channel, dilation=1):
    return nn.Conv3d(in_channel, out_channel, 3,
                  padding=[dilation,dilation,1],
                  dilation=[dilation,dilation,1],
                  bias=True)


# In[3]:


# ASPP
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        
        pyramids = [6, 12, 18, 24]
        
        for i, dilation in enumerate(pyramids):
            self.stages.add_module(
                "aspp{}".format(i),
                conv_dil(in_channels, out_channels, dilation)
            )
        
        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h


# In[4]:


# residual block
class Residual(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super(Residual, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if out_channel is None:
            out_channel = in_channel
        
        self.conv_block = nn.Sequential(
            conv_one(in_channel, out_channel // 2),
            conv_thr(out_channel // 2, out_channel // 2),
            conv_one(out_channel // 2, out_channel)
        )
        
        self.skip_layer = None
        if in_channel != out_channel:
            self.skip_layer = nn.Conv3d(in_channel, out_channel, 1)
        
    def forward(self, x):
        skip = x
        
        out = self.conv_block(x)
        
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        
        out += skip
        
        return out


# In[5]:


# hourglass
class Hourglass(nn.Module):
    def __init__(self, size, channels):
        super(Hourglass, self).__init__()
        self.size = size
        self.channels = channels
        
        self.skip_layers = nn.ModuleList(
            [Residual(
                self.channels
            ) for i in range(self.size)]
        )
        self.low_layers = nn.ModuleList(
            [Residual(
                self.channels
            ) for i in range(self.size)]
        )
        self.mid = Residual(self.channels)
        self.up_layers = nn.ModuleList(
            [Residual(
                self.channels
            ) for i in range(self.size)]
        )
        self.max_pool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        # nearest not working with tuple(1, 2, 2). so trilinear

    def forward(self, x):
        inner = x

        skip_outputs = list()
        for skip, low in zip(self.skip_layers, self.low_layers):
            s = skip(inner)
            skip_outputs.append(s)
            inner = self.max_pool(inner)
            inner = low(inner)
        
        out = self.mid(inner)

        for skip, up in zip(reversed(skip_outputs), reversed(self.up_layers)):
            out = skip + self.upsample(up(out))

        return out


# In[6]:


# model
class Model(nn.Module):
    def __init__(self, features, internal_size=6):
        super(Model, self).__init__()
        self.features = features
        self.internal_size = internal_size
        
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 8, 7, stride=(1, 2, 2),
                      padding=3, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            Residual(8, 16),
        )
        
        self.mid_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            Residual(16, 32),
            Residual(32, self.features)
        )
        
        self.skip_conv = Residual(16, 16)
        
        self.hourglass = Hourglass(self.internal_size, self.features)
        
        self.up_conv = nn.Sequential(
            Residual(self.features, self.features),
            nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
        )
        
        self.aspp = ASPP(self.features, self.features)
        
        self.out_conv = nn.Sequential(
            conv_one(self.features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        init = self.init_conv(x)
        mid = self.mid_conv(init)
        
        hg = self.hourglass(mid)
        
        up = self.up_conv(hg)
        
        out = self.out_conv(up)

        return out
# network structure end -------------------------------------------------

#load model
print("Building model architecture...")
model = Model(64)
print("Loading data...")
pretrained_model = torch.load('data/261.save')
model.load_state_dict(pretrained_model['state'])

model = model.eval()

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

training = False

#read file
root = Tk()
root.withdraw()

filename = askopenfilename(title="Select CT file")
if filename == "":
    exit()

print("Reading file...")

image = 0
if filename.lower().endswith('.nii'):
    image = readNII(filename)

#windowing
image = windowing(image)

#resize to 256x256
image = [Image.fromarray(layer) for layer in image]
image = [layer.resize((256, 256), Image.ANTIALIAS) for layer in image]
image = [np.array(layer) for layer in image]
image = np.asarray(image)

#save input image
#for idx, layer in enumerate(image):
#    img = Image.fromarray(layer)
#    img = img.convert("RGB")
#    img.save("input/{}.png".format(str(idx)))

# test a CT
print("Evaluate CT...")

training = False
torch.set_num_threads(4)

ToTensor = transforms.ToTensor()

with torch.set_grad_enabled(training):
    #ToTensor change [0, 255] value to [0, 1]
    image = ToTensor(image)
    image = image.permute(1, 2, 0)
    
    output = []
            
    for index in range(0, len(image), 3):
        final = 0
        if index+3 > len(image):
            final = index+3 - len(image)
            index = len(image) - 3
                    
        input_cell = image[index:index+3]
        input_cell = input_cell.unsqueeze(0).unsqueeze(0)
        input_cell = input_cell.to(device)
        out_cell = model(input_cell)
        out_cell = out_cell.squeeze()
        if final is 0:
            feed = out_cell.cpu().data.numpy()
            for layer in feed:
                output.append(layer)
        else:
            feed = out_cell.narrow(0, final, 3-final)
            feed = feed.cpu().data.numpy()
            for layer in feed:
                output.append(layer)

    if training:
        raise Exception('There\'s no Training code available.')
    else:
        #threshold
        output = np.asarray(output)
        output = np.ceil(output - 0.81)
        
        directory = "image"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        for idx, layer in enumerate(output):
            img = Image.fromarray(layer*255)
            img = img.convert("RGB")
            img.save(directory + "/{}.png".format(str(idx)))

print("Network evaluation finished.")