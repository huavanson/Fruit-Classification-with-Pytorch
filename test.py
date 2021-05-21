import pickle 
from PIL import Image
import matplotlib.pyplot as plt
from ImageTransform import *
from Network import *
import torch
import tqdm
from torch import nn
import torchvision
import numpy as np


img_file_path ='C:\\Users\\Admin\\Desktop\\pytorch_tranferlearning\\Fruits-Classification\\data\\apple\\apple_0.jpg'
svc = pickle.load(open('model_svc','rb'))
img = Image.open(img_file_path)
vgg = EncoderVGG()
vgg.fine_tune(False)
plt.imshow(img)
plt.show()

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = ImageTransform(resize, mean, std)
img_transformed = transform(img)
img_out = img_transformed.unsqueeze(0)
img_encode = vgg(img_out)
img_encode = torch.detach(img_encode).numpy()
        
print(svc.predict(img_encode))