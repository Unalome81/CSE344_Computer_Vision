import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import wandb

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

import torch.optim as optim

from torchvision.models import resnet18, ResNet18_Weights

import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import ConcatDataset

def Extract_Images(folder):
    paths = []
    for file_name in os.listdir(folder):
        img_path = os.path.join(folder, file_name)
        paths.append(img_path)

    return paths

# Building a custom dataset to preprocess the data and store it as tensors using its file paths
class CustomImageDataset(Dataset):
    def __init__(self, Image_Paths, Mask_Paths, transform_rgb=None, transform_gray = None):
        self.Image_Paths = Image_Paths
        self.Mask_Paths = Mask_Paths
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray

    def __len__(self):
        return len(self.Image_Paths)

    def __getitem__(self, idx):
        image = Image.open(self.Image_Paths[idx]).convert('RGB')
        if self.transform_rgb:
            image = self.transform_rgb(image)

        mask = Image.open(self.Mask_Paths[idx]).convert('L')

        if self.transform_gray:
            image = self.transform_gray(mask)
    
        return image, mask


image_folder = "IDD20K_II\image_archive"
mask_folder = "IDD20K_II\mask_archive"

Image_paths = Extract_Images(image_folder)
Mask_paths = Extract_Images(mask_folder)

image_height = 512
image_width = 512

transform_rgb = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])  

transform_gray = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])  

Driving_Dataset = CustomImageDataset(Image_paths, Mask_paths, transform_rgb, transform_gray)

Driving_Loader = DataLoader(Driving_Dataset, batch_size = 64, shuffle=True)