

# Commented out IPython magic to ensure Python compatibility.
#all imports / data dependencies

#general dependencies
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#sklearn imports
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelBinarizer

#pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms

import datetime
###############################################################################################
# CHANGE THIS FILE PATH TO A FOLDER WHERE DATA AND TRAINED MODEL ARE
FILE_PATH = 'D:/ML/'
##############################################################################################

class brick_dataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.data = X
        self.target = y
        self.transform = transform
        
        if torch.cuda.is_available():
            print("Data placed in GPU memory")
            self.data = self.data.cuda()
            
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        
        if torch.cuda.is_available():
            return x.cuda(), y.cuda()
        
        return x,y
        
    def __len__(self):
        return len(self.data)


class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out1 = nn.Dropout(p=.06)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out2 = nn.Dropout(p=.06)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out3 = nn.Dropout(p=.07)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out4 = nn.Dropout(p=.1)
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out5 = nn.Dropout(p=.6)
        self.fc1 = nn.Linear(6 * 6 * 1024, 3600)
        self.bn1 = nn.BatchNorm1d(3600)

        self.fc4 = nn.Linear(3600, 5)

        print("model device:", self.device)




    def forward(self, x):

        out = self.layer1(x)
        out = self.drop_out1(out)
        out = self.layer2(out)
        out = self.drop_out2(out)
        out = self.layer3(out)
        out = self.drop_out2(out)
        out = self.layer4(out)
        out = self.drop_out4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out5(out)
        out = nn.functional.relu(self.bn1(self.fc1(out)))

        out = self.fc4(out)

        return out



def test_model(Images):
    model = conv_net()
    model.load_state_dict(torch.load(FILE_PATH + "trained_model.pt")['model_state_dict'])
    if model.use_cuda:
        model.cuda()
    model.eval()

    X_val_t = torch.tensor(Images, dtype=torch.float32) #divide by 255 to scale between [0,1]
    X_val_t = X_val_t.permute(0,3,1,2)


    val_transforms = transforms.Compose([
                                            transforms.ToPILImage(mode='RGB'),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5628, 0.4763, 0.4404],[0.1937, 0.1930, 0.2082])])

    y_val_t = torch.tensor(np.zeros(X_val_t.shape[0]))

    val_dataset = brick_dataset(X_val_t, y_val_t, transform = val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle=False)
    with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0

                predicted_list = []
                for imgs,labs in val_loader:
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_list.append(predicted.item())

    return predicted_list
##################################################################################################
#IMPORTANT
#CHANGE THIS VARIABLE TO YOUR FILE PATH THAT CONTAINS THE  Training data, and Labels
##################################################################################################
Images = np.load(FILE_PATH+"Images.npy")/255
#Labels = np.load(FILE_PATH+"Labels.npy")


predicted = test_model(Images)
print(predicted)
