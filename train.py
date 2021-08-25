

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


def train_model(Images, Labels):

    X_train = torch.tensor(Images, dtype=torch.float32) #divide by 255 to scale between [0,1]
    y_train = torch.tensor(Labels.T, dtype=torch.long)

    #pytorch requires set of images be defined as N x C x H x W, convert other np arrays to tensors
    #permute channels to get proper shape
    X_train_t = X_train.permute(0,3,1,2)
    y_train_t = y_train
    #split training data further into training/validation data
    num_splits = 4
    kf = StratifiedKFold(n_splits = num_splits)
    indices = []
    for train_index, test_index in kf.split(X_train, y_train):
        indices.append([train_index,test_index])


    #just looking at first fold for now
    X_train_cv_t = X_train_t[indices[0][0]].contiguous()
    y_train_cv_t = torch.tensor(y_train_t[indices[0][0]].numpy(), dtype=torch.long)
    X_val_cv_t = X_train_t[indices[0][1]].contiguous()
    y_val_cv_t = torch.tensor(y_train_t[indices[0][1]].numpy(), dtype=torch.long)

    X_train_cv_t.shape, X_val_cv_t.shape, np.unique(y_train_cv_t.numpy(), return_counts = True)

    #Need to find means and standard deviations of the current training data for normalization transform
    #need to reshape data to N X C x W*H
    X_train_cv_t2 = X_train_t.permute(1,2,3,0).contiguous()
    train_transforms = transforms.Compose([
                                            transforms.ToPILImage(mode='RGB'),
                                            transforms.RandomApply(
                                                torch.nn.ModuleList([
                                                     transforms.RandomAffine(45),
                                                     transforms.RandomRotation(45),
                                                     transforms.RandomHorizontalFlip(p=0.25),
                                                     transforms.ColorJitter(),
                                                     transforms.RandomPerspective(p=1)
                                            ])),
                                            transforms.ToTensor(),
                                            transforms.Normalize(X_train_cv_t2.view(3,-1).mean(dim=1),X_train_cv_t2.view(3,-1).std(dim=1))])


    val_transforms = transforms.Compose([
                                            transforms.ToPILImage(mode='RGB'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(X_train_cv_t2.view(3,-1).mean(dim=1),X_train_cv_t2.view(3,-1).std(dim=1))])





    #data set class definition



    train_dataset = brick_dataset(X_train_cv_t, y_train_cv_t, transform=train_transforms)
    val_dataset = brick_dataset(X_val_cv_t, y_val_cv_t, transform=train_transforms)

######################################################################################################################################
################################################# PARAMETERS #########################################################################
    # Setup model and pertinent parameters.
    n_epochs = 500
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    model = conv_net()
    lr = 1e-3
    weight_decay = 0 #3e-5
    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    max_lr = 3e-3
    steps_per_epoch = len(train_loader)

    loss_fn = nn.CrossEntropyLoss()

 ################################################################################################################################
    if model.use_cuda:
      if(torch.cuda.device_count() > 1):
          print("Using data parallel for training model")
          model = nn.DataParallel(model)
    model = model.to(model.device)

    total_step = len(train_loader)

    training_loss_list = []
    training_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Criteria for early stopping.

    # Number of consecutive epochs before validation loss is checked.
    val_rate = 1

    # Number of validation loss checks in which there is no validation loss
    # improvement before early stopping is performed.
    patience = 100

    val_loss_min = float("inf")
    val_acc_max = -1
    val_no_improvement = 0
    keepTraining = True


    start_time = datetime.datetime.now()

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            training_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # Track the training accuracy.
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            training_acc_list.append(correct / total)

            if (i + 1) % 10 == 0:
                print('[{}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                      'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


        if (epoch+1)%val_rate == 0:
            # Every `val_rate` epochs, compute average validation
            # loss and accuracy.

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0

                for imgs, labels in val_loader:
                    outputs = model(imgs)
                    val_loss += loss_fn(outputs,labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                # Compute average validation loss and accuracy.
                val_loss = val_loss / len(val_loader)
                val_acc = (correct / total) * 100

                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)

                print('\n\n[{}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                      'Validation Loss, Epoch {}: {}%'.format(epoch+1, val_loss))
                print('[{}]'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                      'Validation Accuracy, Epoch {}: {}%\n'.format(epoch+1, val_acc))

                if val_acc > val_acc_max:
                    # Average validation loss improved!
                    val_no_improvement = 0
                    val_acc_max = val_acc

                else:
                    # Average validation loss did not improve.
                    val_no_improvement += 1
                    if val_no_improvement == patience:
                        # Patience was exhausted; start early stopping process.
                        keepTraining = False


                print("Number of consecutive epochs without accuracy improvement:",
                      val_no_improvement)
                print("\n\n")

        if keepTraining == True:
            model.train()

        else:
            print('\n\nEarly stopping at epoch {}.'.format(epoch+1))
            # Break out of training loop.
            break;

    end_time = datetime.datetime.now()
    total_time = end_time-start_time

    seconds_in_day = 24*60*60

    # Compute total training time in the format of (minutes, seconds).
    total_train_time = divmod(total_time.days *
                              seconds_in_day + total_time.seconds, 60)
    print("\nTotal training time:", total_train_time[0], "minutes,",
          total_train_time[1], "seconds.")

    t_dataset = brick_dataset(X_train_cv_t, y_train_cv_t, transform=val_transforms)
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

    model.eval()
    for i, loader in enumerate([t_loader, train_loader, val_loader]):
      with torch.no_grad():
        correct = 0
        total = 0
        for imgs, labels in loader:
          outputs = model(imgs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

        if i == 0:
          print('Final Training (No Transform) Accuracy: {} %'.format((correct / total) * 100))
        elif i == 1:
          print('Final Training (With Transform) Accuracy: {} %'.format((correct / total) * 100))
        else:
          print('Final Validation Accuracy: {} %'.format((correct / total) * 100))

    print('Finished run!')

    # Save current model.
    print('\n\nSaving current model...')
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': loss,
                'training_loss_list': training_loss_list,
                'training_acc_list': training_acc_list,
                'val_loss': val_loss,
                'val_loss_list': val_loss_list,
                'val_acc_list': val_acc_list,
                'total_train_time': total_train_time,
                }, '/content/drive/MyDrive/Colab Notebooks/conv_net_model_24.pt')
    print('Successfully saved current model.')


##################################################################################################
#IMPORTANT
#CHANGE THIS VARIABLE TO YOUR FILE PATH THAT CONTAINS THE  Training data, and Labels
file_path = 'c:/Users/Brit Chesley/Documents/Classes/Fall_2020/Machine_Learning/brick_images/'
##################################################################################################
Images = np.load(file_path+"Images.npy")/255
Labels = np.load(file_path+"Labels.npy")
model = conv_net()
train_model(Images, Labels)
