import sys
import os
import re
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import class_weight
import seaborn as sns

waveforms_total = []
# number of waveforms per event
n = 3
# True for required choosing method
KNN = False
MEAN_VARIANCE = False
VARIANCE = True
DM_CHOOSE = False
# number of minutes since seismic event accrued (data getting into neural network)
split_time = 10


class SeismicNet(nn.Module):

    def __init__(self):
        super(SeismicNet, self).__init__()
        self.I_layer = nn.Sequential(
            # 1X3X900 -> 5X3X447
            nn.Conv2d(1, 5, kernel_size=(3, 10), stride=(1, 2), padding=1, padding_mode='zeros'),
            nn.ReLU(),

            #nn.BatchNorm2d(5),
            #nn.ReLU(),
            # 5X3X297 -> 5X3X99
            #nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            #nn.BatchNorm2d(5),

            # 5X3X447 -> 10X3X224
            nn.Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 2), padding=1, padding_mode='zeros'),
            nn.ReLU(),

            #nn.BatchNorm2d(10),
            #nn.ReLU(),
            # 20X3X60 -> 20X3X60
            #nn.BatchNorm2d(20),

            # 10X3X224 -> 10X3X112
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.BatchNorm2d(10))
            # 10X3X112 -> 5X1X55
            #nn.Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 2), padding=0, padding_mode='zeros'),
            #nn.ReLU(),

            # 40X1X5 -> 40X1X5
            #nn.ReLU(),

            # 5X1X55 -> 5X1X18
            #nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            #nn.BatchNorm2d(5))
            # 40X1X4 -> 20X1X1
            #nn.Conv2d(40, 20, kernel_size=(1, 3), stride=(1, 2)),
            #nn.BatchNorm2d(20))

        self.Guy_layer = nn.Sequential(
            nn.Linear(3360, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.Linear(400, 9),
        )

        self.II_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(90, 9),
            nn.ReLU(),)
            #nn.Dropout(p=0.1),
            #nn.Linear(10, 1))
            #nn.ReLU(),
            #nn.Dropout(p=0.1),
            #nn.Linear(10, 1))
            #nn.Softmax(dim=1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.I_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.Guy_layer(x)
        # x = self.II_layer(x)
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
    return running_loss / train_size


def test_step(model, testloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct, check = 0, 0

    nb_classes = 9
    confusion_matrix = np.zeros((nb_classes, nb_classes))


    with tqdm(total=len(testloader)) as pbar:
        for i, data in enumerate(testloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            loss = criterion(outputs.squeeze(), labels)
            # print statistics
            running_loss += loss.item()
            # regression accuracy
            '''
            lower_bound = labels - 0.01
            upper_bound = labels + 0.01
            temp = [lower_bound[i] <= outputs[i] <= upper_bound[i] for i in range(len(labels))]
            correct += np.sum(temp)
            lower_bound_0 = labels - 0.1
            upper_bound_0 = labels + 0.1
            temp_0 = [lower_bound_0[i] <= outputs[i] <= upper_bound_0[i] for i in range(len(labels))]
            check += np.sum(temp_0)
            '''
            # classification accuracy
            temp = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            correct += np.sum(temp)

            temp_0 = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            check += np.sum(temp_0)

            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)

    plt.figure(figsize=(15, 10))

    class_names = [i for i in range(1,10)]
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return running_loss / test_size, correct*100/test_size, check*100/test_size


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
            # regression accuracy
            '''
            lower_bound = labels - 0.01
            upper_bound = labels + 0.01
            temp = [lower_bound[i] <= outputs[i] <= upper_bound[i] for i in range(len(labels))]
            correct += np.sum(temp)
            lower_bound_0 = labels - .1
            upper_bound_0 = labels + .1
            temp_0 = [lower_bound_0[i] <= outputs[i] <= upper_bound_0[i] for i in range(len(labels))]
            check += np.sum(temp_0)
            '''

            # classification accuracy
            temp = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            correct += np.sum(temp)

            temp_0 = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            check += np.sum(temp_0)

            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)
    return running_loss / train_size, correct*100/train_size, check*100/train_size


def read_waveforms(waveforms, split_time):
    waveforms = [line for line in waveforms if line != '\n']
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        waveforms_total.append(torch.as_tensor([float(elem) for elem in waveforms_list[:(split_time+5)*60]], dtype=torch.float))
    return


if KNN is True:
    # path of data while picking waveforms using the knn method
    dir_path = os.path.join(os.getcwd(), f"knn_pick_{n}")
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_knn_{n}.pt"
    # path of labels tensor
    LABELS_PATH = f"labels_tensor_knn_{n}.pt"
if MEAN_VARIANCE is True:
    # path of data while picking waveforms using the mean variance method
    dir_path = os.path.join(os.getcwd(), f"mean_variance_pick_{n}")
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_mean_variance_{n}.pt"
    # path of labels tensor
    LABELS_PATH = f"labels_tensor_mean_variance_{n}.pt"
if VARIANCE is True:
    # path of data while picking waveforms using the variance method
    dir_path = os.path.join(os.getcwd(), f"variance_pick_{n}")
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_variance_{n}.pt"
    # path of labels tensor
    LABELS_PATH = f"labels_tensor_variance_{n}.pt"
if DM_CHOOSE is True:
    # path of data while picking waveforms using the dm method
    dir_path = os.path.join(os.getcwd(), f"dm_pick_{n}")
    # path of waveforms tensors
    DATA_PATH = f"waveforms_tensor_dm_{n}.pt"
    # path of labels tensor
    LABELS_PATH = f"labels_tensor_dm_{n}.pt"

if not os.path.exists(LABELS_PATH):
    fd_labels = open("labels.txt", 'r')
    labels = fd_labels.readlines()
    labels = labels[0].split(' ')
    labels = np.array([float(item) for item in labels[:-1]], dtype='float32')
    fd_labels.close()

    input_files = np.array(os.listdir(dir_path))
    indices = np.array([int(re.findall('[0-9]+', file)[0]) for file in input_files])
    chosen_labels = labels[indices]
    labels_tensor = torch.from_numpy(chosen_labels)
    torch.save(labels_tensor, LABELS_PATH)
else:
    labels_tensor = torch.load(LABELS_PATH)

if not os.path.exists(DATA_PATH):
    file_path = [os.path.join(dir_path, input_file) for input_file in input_files]
    input_len = (split_time + 5) * 60
    for file in file_path:
        fd = open(file, 'r')
        lines = fd.readlines()
        read_waveforms(lines, split_time)
        fd.close()
    waveforms_tensor = [F.pad(item, (int(np.floor((input_len - item.size()[0]) / 2)), int(np.ceil((input_len - item.size()[0]) / 2))))
                        for item in waveforms_total]
    waveforms_tensor = torch.reshape(torch.stack(waveforms_tensor, 0), (-1, n, input_len))
    torch.save(waveforms_tensor, DATA_PATH)
else:
    waveforms_tensor = torch.load(DATA_PATH)


def normalize_data(waveforms_tensor):
    temp_tensor = torch.zeros(waveforms_tensor.size())
    temp_tensor[:, :, 1:] = torch.clone(waveforms_tensor[:, :, :-1])
    res_tensor = waveforms_tensor - temp_tensor

    min_value = torch.min(res_tensor[:, :, 0])
    max_value = torch.max(res_tensor[:, :, 0])
    res_tensor[:, :, 0] -= min_value
    res_tensor[:, :, 0] /= (max_value - min_value)
    return res_tensor


def normalize_labels(labels_tensor):
    #normalization for regression
    #labels_tensor /= 10
    #normalization for classification
    labels_tensor = torch.round(labels_tensor).long()
    labels_tensor -= 1
    return labels_tensor


normelized_waveforms_tensor = normalize_data(waveforms_tensor)
normelized_labels_tensor = normalize_labels(labels_tensor)

epochs = 50
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#criterion = nn.MSELoss(reduction="sum")
labels_arr = normelized_labels_tensor.numpy()
weight = class_weight.compute_class_weight('balanced', np.unique(labels_arr), labels_arr).astype(np.float32)
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight))
# criterion = nn.CrossEntropyLoss()
model = SeismicNet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_train, x_test, y_train, y_test = train_test_split(normelized_waveforms_tensor, normelized_labels_tensor,
                                                    test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
x_train = x_train[:20000]
x_train = torch.FloatTensor(x_train)
y_train = y_train[:20000]

# plt.hist(y_train[:1000], bins=10)
# plt.show()
x_test = x_test
x_test = torch.FloatTensor(x_test)
y_test = y_test

train_size = x_train.size()[0]
test_size = x_test.size()[0]


train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

batch_size = 32
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

for epoch in range(epochs):  # loop over the dataset multiple times
    train_loss = train_step(model, trainloader, criterion, optimizer, epoch)
    _, acc_t, check_t = train_check(model, trainloader, criterion, epoch)
    test_loss, accuracy, check = test_step(model, testloader, criterion, epoch)
    print('\n[%d] Train loss: %.3f Test loss: %.3f accuracy: %.3f %s better accuracy: %.3f %s' % (epoch, train_loss,
                                                                                                  test_loss, accuracy,
                                                                                                  '%', check, '%'))
    print(f"acc train : {acc_t}, better acc_t: {check_t}")


print('Finished Training')

