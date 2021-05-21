from sklearn.svm import SVC
import torch
from torch import nn
import torchvision
from PIL import Image
import os 
import glob

class Dataloader(nn.Module):
    def __init__(self , file_path ,encoder, transform=None):

        super(Dataloader,self).__init__()
        self.file_path = file_path
        self.encoder = encoder
        self.transform = transform
    def __len__(self) :
        return len(self.file_path) 



    def __getitem__(self, index) :
        img_path = self.file_path[index]
        img = Image.open(img_path)
        img_transform = self.transform(img)
        img_transform = img_transform.unsqueeze(0)
        img_encoder = self.encoder(img_transform)
        img_encoder = torch.detach(img_encoder).numpy()
        img_encoder = img_encoder.reshape(img_encoder.shape[1])
        
        if 'apple' in img_path :
            labels = 0
        if 'banana' in img_path :
            labels = 1
        if 'grape' in img_path :
            labels = 2
        if 'mango' in img_path :
            labels = 3
        if 'orange' in img_path :
            labels = 4
        if 'pear' in img_path :
            labels = 5
        if 'pineapple' in img_path :
            labels = 6
        if 'tangerine' in img_path :
            labels = 7
        if 'tomato' in img_path :
            labels = 8
        if 'watermelon' in img_path :
            labels = 9                                
        return img_encoder , labels 



def create_dataset_path(file_path):
    files_path = []
    for ix in os.listdir(file_path):
        for yx in os.listdir(file_path+'\\'+ix):
            files_path.append(os.path.join(file_path+'\\'+ix,yx))
    return files_path 