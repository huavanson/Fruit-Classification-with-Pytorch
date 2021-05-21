import glob 
import os 
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    
    def __call__(self, img):
        return self.data_transform(img)
