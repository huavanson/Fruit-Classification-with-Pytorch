from Network import *
from ImageTransform import *
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data_path = 'C:\\Users\\Admin\\Desktop\\pytorch_tranferlearning\\Fruits-Classification\\data\\apple'
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
alexnet = EncoderAlexNet()
alexnet.fine_tune(False)
target_name = ['apple','banana','grape','mango','orange','pear','pineapple','tangerine','tomato','watermelon']

images_encode_alexnet= pickle.load(open('train_encoded_images_alexnet','rb'))
labels_alexnet = pickle.load(open('label_images_alexnet','rb'))
images_encode_vgg= pickle.load(open('train_encoded_images','rb'))
labels_vgg = pickle.load(open('label_images_vgg','rb'))

svc = pickle.load(open('model_svc_alexnet','rb'))

svc_vgg = pickle.load(open('model_svc','rb'))


def accuracy_confusionmatrix_report(model, images, labels) :
    
    target_names = ['apple','banana','grape','mango','orange','pear',
                    'pineapple','tangerine','tomato','watermelon']

    predict = model.predict(images) 
    confusion_mtx = confusion_matrix(labels, predict) 
   
    return accuracy_score(labels,predict) , confusion_mtx , classification_report(labels,predict, target_names=target_names)
def confusionmatrix_visualize(confusion_mtx) :
    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix_ALEXNET")
    plt.show()        
def report_classification(report) :
    print(report)
     

if __name__ == '__main__':        
    accuracyscore, matrix ,report = accuracy_confusionmatrix_report(svc,images_encode_alexnet,labels_alexnet)
    print(accuracyscore)
    confusionmatrix_visualize(matrix)
    report_classification(report)