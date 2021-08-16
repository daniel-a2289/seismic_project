import sys
import os
import re
import subprocess
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import torchvision
import scipy
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

waveforms_total = []
# number of waveforms per event
n = 3
# True for wanted choosing method
KNN = False
MEAN_VARIANCE = False
VARIANCE = True
DM_CHOOSE = False


class SeismicNet(nn.Module):

    def __init__(self):
        super(SeismicNet, self).__init__()
        self.I_layer = nn.Sequential(
            # 1X3X3901 -> 10X3X969
            nn.Conv2d(1, 10, kernel_size=(3, 30), stride=(1, 4), padding=1, padding_mode='circular'),
            #  nn.BatchNorm2d(18000),
            # 10X3X969 -> 10X3X121
            nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
            # 10X3X121 -> 20X3X60
            nn.Conv2d(10, 20, kernel_size=(3, 4), stride=(1, 2), padding=1, padding_mode='circular'),
            # 20X3X60 -> 20X3X60
            nn.BatchNorm2d(20),
            # 20X3X60 -> 20X3X15
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            # 20X3X15 -> 40X1X5
            nn.Conv2d(20, 40, kernel_size=(3, 3), stride=(1, 3), padding=0, padding_mode='circular'),
            # 40X1X5 -> 40X1X5
            nn.ReLU(),
            # 40X1X5 -> 40X1X4
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)),
            # 40X1X4 -> 20X1X1
            nn.Conv2d(40, 20, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(20))

        self.II_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(20, 20),
            nn.Dropout(p=0.1),
            nn.Linear(20, 10),
            nn.Dropout(p=0.1),
            nn.Linear(10, 1))
            # nn.Softmax(dim=1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.I_layer(x)
        x = self.II_layer(torch.flatten(x, start_dim=1))
        return x


def train_step(model, trainloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(trainloader)) as pbar:
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)
    return running_loss / (len(trainloader) * batch_size)


def test_step(model, testloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct, check = 0, 0
    with tqdm(total=len(testloader)) as pbar:
        for i, data in enumerate(testloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            # print statistics
            running_loss += loss.item()
            # lower_bound = labels - 0.05*labels
            # upper_bound = labels + 0.05*labels
            lower_bound = labels - 0.1
            upper_bound = labels + 0.1
            temp = [lower_bound[i] <= outputs[i] <= upper_bound[i] for i in range(len(labels))]
            correct += np.sum(temp)
            lower_bound_0 = labels - 1
            upper_bound_0 = labels + 1
            temp_0 = [lower_bound_0[i] <= outputs[i] <= upper_bound_0[i] for i in range(len(labels))]
            check += np.sum(temp_0)
            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)
    return running_loss / (len(testloader) * batch_size), correct*100/(len(testloader) * batch_size), \
           check*100/(len(testloader) * batch_size)


def train_check(model, trainloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct, check = 0, 0
    with tqdm(total=len(trainloader)) as pbar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
            lower_bound = labels - 0.1
            upper_bound = labels + 0.1
            temp = [lower_bound[i] <= outputs[i] <= upper_bound[i] for i in range(len(labels))]
            correct += np.sum(temp)
            lower_bound_0 = labels - 1
            upper_bound_0 = labels + 1
            temp_0 = [lower_bound_0[i] <= outputs[i] <= upper_bound_0[i] for i in range(len(labels))]
            check += np.sum(temp_0)
            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)
    return running_loss / (len(trainloader) * batch_size), correct*100/(len(trainloader) * batch_size), \
           check*100/(len(trainloader) * batch_size)

def read_waveforms(waveforms, max_len):
    waveforms = [line for line in waveforms if line != '\n']
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        waveforms_total.append(torch.as_tensor([float(elem) for elem in waveforms_list], dtype=torch.float))
        if max_len < len(waveforms_list):
            max_len = len(waveforms_list)
    return max_len


if KNN is True:
    dir_path = os.path.join(os.getcwd(), f"knn_pick_{n}")
    input_files = np.array(os.listdir(dir_path))
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_knn_{n}.pt"
if MEAN_VARIANCE is True:
    dir_path = os.path.join(os.getcwd(), f"mean_variance_pick_{n}")
    input_files = np.array(os.listdir(dir_path))
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_mean_variance_{n}.pt"
if VARIANCE is True:
    dir_path = os.path.join(os.getcwd(), f"variance_pick_{n}")
    input_files = np.array(os.listdir(dir_path))
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_variance_{n}.pt"
if DM_CHOOSE is True:
    dir_path = os.path.join(os.getcwd(), f"dm_pick_{n}")
    input_files = np.array(os.listdir(dir_path))
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_dm_{n}.pt"

fd_labels = open("labels.txt", 'r')
labels = fd_labels.readlines()
labels = labels[0].split(' ')
labels = np.array([float(item) for item in labels[:-1]], dtype='float32')
fd_labels.close()

indices = np.array([int(re.findall('[0-9]+', file)[0]) for file in input_files])
labels_files = labels[indices]

if not os.path.exists(DATA_PATH):
    file_path = [os.path.join(dir_path, input_file) for input_file in input_files]
    curr_len = 0
    for file in file_path:
        fd = open(file, 'r')
        lines = fd.readlines()
        curr_len = read_waveforms(lines, curr_len)
        fd.close()
    waveforms_tensor = [F.pad(item, (int(np.floor((curr_len-item.size()[0])/2)), int(np.ceil((curr_len-item.size()[0])/2))))
                        for item in waveforms_total]
    waveforms_tensor = torch.reshape(torch.stack(waveforms_tensor, 0), (-1, n, curr_len))
    torch.save(waveforms_tensor, DATA_PATH)
else:
    waveforms_tensor = torch.load(DATA_PATH)


'''
class SeismicNet(nn.Module):
    def __init__(self):
        super(SeismicNet, self).__init__()
        self.I_layer = nn.conv2d(points, 17977, (3, 240), stride=(1, 10), padding=1, padding_mode='circular')  # 5xpoints to 3x3
        self.II_layer = nn.BatchNorm2d(17977)  # 3x3 to 3x1
        self.III_layer = nn.MaxPool2d((1, 8), stride=(1, 8))  # 1x3 to 1x1
        self.IV_layer = nn.conv2d(2248, 447, (3, 20), stride=(1, 5), padding=1, padding_mode='circular')  # 3x3 to 3x1
        self.V_layer = nn.BatchNorm2d(447)  # 1x3 to 1x1
        self.VI_layer = nn.MaxPool2d((1, 4), stride=(1, 4), padding=(0, 1))  # 3x3 to 3x1
        ## stopped here
        self.VII_layer = nn.conv2d(113, 17977, (3, 240), stride=(1, 10), padding=1, padding_mode='circular')  # 1x3 to 1x1
        self.VIII_layer = nn.BatchNorm2d(100)  # 3x3 to 3x1
        self.IX_layer = nn.MaxPool2d(3, stride=2)  # 1x3 to 1x1
        self.X_layer = nn.conv2d(points, 3)  # 3x3 to 3x1
        self.XI_layer = nn.BatchNorm2d(100)  # 1x3 to 1x1
        self.XII_layer = nn.AvgPool2d(3, stride=2)  # 3x3 to 3x1
        self.XIII_layer = nn.Softmax(dim=1)  # 1x3 to 1x1

    def forward(self, x):
        x = self.I_layer(x.T)
        x = self.II_layer(x)
        x = self.III_layer(x.T)
        x = self.IV_layer(x)
        x = self.V_layer(x.T)
        x = self.VI_layer(x)
        x = self.VII_layer(x.T)
        x = self.VIII_layer(x)
        x = self.IX_layer(x.T)
        x = self.X_layer(x)
        x = self.XI_layer(x.T)
        x = self.XII_layer(x)
        x = self.XIII_layer(x.T)

        return x
'''

epochs = 30
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction="sum")
model = SeismicNet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_train, x_test, y_train, y_test = train_test_split(waveforms_tensor, torch.from_numpy(labels_files),
                                                    test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
x_train = x_train[:1000]
x_train = torch.FloatTensor(x_train)
y_train = y_train[:1000]
# plt.hist(y_train[:1000], bins=10)
# plt.show()
x_test = x_test[:18000]
x_test = torch.FloatTensor(x_test)
y_test = y_test[:18000]


train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

batch_size = 128
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

'''
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_time = time.time()
    for i in range(int(num_times)):
        work_data = torch.from_numpy(acce[points*i : points*(i+1)])
        distance = torch.norm(torch.from_numpy(pos[points * (i + 1)] - pos[points * i]))
        distance = distance.float()
        work_data = work_data.float()
        output = model(work_data)
        loss = criterion(output, distance)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
    log = "Epoch: {} | Loss: {:.4f} |".format(epoch, running_loss/num_times)
    epoch_time = time.time() - epoch_time
    log += " Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)

'''

for epoch in range(epochs):  # loop over the dataset multiple times
    train_loss = train_step(model, trainloader, criterion, optimizer, epoch)
    _, acc_t, check_t = train_check(model, trainloader, criterion, epoch)
    test_loss, accuracy, check = test_step(model, testloader, criterion, epoch)
    print('\n[%d] Train loss: %.3f Test loss: %.3f accuracy: %.3f %s better accuracy: %.3f %s' % (epoch, train_loss,
                                                                                                  test_loss, accuracy,
                                                                                                  '%', check, '%'))
    print(f"acc train : {acc_t}, better acc_t: {check_t}")
print('Finished Training')

