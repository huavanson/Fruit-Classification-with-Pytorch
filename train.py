from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import tqdm
from torch import nn
import torchvision
from PIL import Image
from Network import *
from ImageTransform import *
from data import *
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
fine_tune = False
alexnet = EncoderAlexNet()
alexnet.fine_tune(fine_tune)
vgg = EncoderVGG()
vgg.fine_tune(fine_tune)
svc = SVC(random_state = 42)
data_path = 'C:\\Users\\Admin\\Desktop\\pytorch_tranferlearning\\Fruits-Classification\\data'
def main(encoder, data_path):

    files_data_path = create_dataset_path(data_path)
    dataloader = Dataloader(file_path=files_data_path,encoder=encoder,transform=ImageTransform(resize,mean,std))

    images = []
    nhans = []
    for i in tqdm(range(dataloader.__len__())):
        #image = image.to(device)
        #nhan = nhan.to(device)
        try:
            image , nhan = dataloader.__getitem__(i)
            images.append(image)
            nhans.append(nhan)
        except :
            continue    
    with open( "train_encoded_images", "wb" ) as pickle_f:
        pickle.dump(images, pickle_f )    
    with open( "label_images_vgg", "wb" ) as pickle_f:
        pickle.dump(nhans, pickle_f )      
    x_train , x_val , y_train , y_val = train_test_split(images,nhans,
                                                        test_size=0.2,
                                                        random_state=0)
    svc.fit(x_train,y_train)
    predict = svc.predict(x_val)
    acc_score = accuracy_score(y_val,predict)
    print(acc_score)
    #with open( "model_svc_alexnet", "wb" ) as pickle_f:
        #pickle.dump(svc, pickle_f ) 

main(vgg,data_path)