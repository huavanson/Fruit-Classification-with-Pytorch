import torch
from torch import nn
import torchvision 
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

class EncoderVGG(nn.Module) :
    def __init__(self):
        
        super(EncoderVGG, self).__init__()
        #self.enc_image_size = encoded_image_size

        vgg19 = torchvision.models.vgg19(pretrained=True)  # pretrained ImageNet vgg19

            # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg19.children())[:-2]
        self.vgg19 = nn.Sequential(*modules)



    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.vgg19(images)  # (1 , 512 , 7 ,7 )
        out = out.mean(dim=(-2, -1)) # (1 , 512)

        return out  
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.vgg19.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.vgg19.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune    
class EncoderAlexNet(nn.Module):
    def __init__(self) :
      
        super(EncoderAlexNet, self).__init__()
        #self.enc_image_size = encoded_image_size

        alexnet = torchvision.models.alexnet(pretrained=True)  # pretrained alxenet

            # Remove linear and pool layers (since we're not doing classification)
        modules = list(alexnet.children())[:-2]
        self.alexnet = nn.Sequential(*modules)




    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.alexnet(images) 
        out = out.mean(dim=(-2, -1))
        return out  
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.alexnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.alexnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune    
