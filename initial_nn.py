
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

waveforms_total = []
# number of waveforms per event
n = 1
# True for required choosing method
KNN = True
MEAN_VARIANCE = False
VARIANCE = False
DM_CHOOSE = False
# number of minutes since seismic event accrued (data getting into neural network)
split_time = 10


class SeismicNet(nn.Module):

    def __init__(self):
        super(SeismicNet, self).__init__()
        self.I_layer = nn.Sequential(
            # 1X3X900 -> 5X3X447
            nn.Conv2d(1, 1, kernel_size=(3, 10), stride=(1, 2), padding=1, padding_mode='zeros'),
            nn.ReLU(),

            # 5X3X447 -> 10X3X224
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 2), padding=1, padding_mode='zeros'),
            nn.ReLU(),

            # 10X3X224 -> 10X3X112
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.BatchNorm2d(1))

        # for n=1
        self.option_one = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(112, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
        )
        # for n=2
        self.option_two = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(224, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
        )
        # for n=3
        self.option_three = nn.Sequential(
            nn.Dropout(p=0.2),

            nn.Linear(336, 40),
            nn.ReLU(),
            #nn.Linear(1000, 400),
            #nn.ReLU(),
            nn.Linear(40, 7),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.I_layer(x)
        x = torch.flatten(x, start_dim=1)
        if n == 1:
            x = self.option_one(x)
        if n == 2:
            x = self.option_two(x)
        if n == 3:
            x = self.option_three(x)
        return x


def train_step(model, trainloader, criterion, optimizer, epoch, train_size):
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


def test_step(model, testloader, criterion, epoch, test_size):
    model.eval()
    running_loss = 0.0
    acc_1, acc_2, acc_3 = 0, 0, 0

    nb_classes = 7
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

            # classification accuracy
            temp = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            acc_1 += np.sum(temp)

            temp_1 = [any(torch.argsort(outputs[i])[-2:] == labels[i]) for i in range(len(labels))]
            acc_2 += np.sum(temp_1)

            temp_0 = [any(torch.argsort(outputs[i])[-3:] == labels[i]) for i in range(len(labels))]
            acc_3 += np.sum(temp_0)

            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)

    # heat map visualization
    '''
    plt.figure(figsize=(15, 10))

    class_names = [i for i in range(1, 8)]
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    '''

    return running_loss / test_size, acc_1*100/test_size, acc_2*100/test_size, acc_3*100/test_size


def train_check(model, trainloader, criterion, epoch, train_size):
    model.eval()
    running_loss = 0.0
    acc_1, acc_2, acc_3 = 0, 0, 0
    with tqdm(total=len(trainloader)) as pbar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()

            # classification accuracy
            temp = [outputs[i].argmax() == labels[i] for i in range(len(labels))]
            acc_1 += np.sum(temp)

            temp_0 = [any(torch.argsort(outputs[i])[-2:] == labels[i]) for i in range(len(labels))]
            acc_2 += np.sum(temp_0)

            temp_1 = [any(torch.argsort(outputs[i])[-3:] == labels[i]) for i in range(len(labels))]
            acc_3 += np.sum(temp_1)

            pbar.set_description(f"Epoch: {epoch} current loss: {loss.item()}")

            pbar.update(1)
    return running_loss / train_size, acc_1*100/train_size, acc_2*100/train_size, acc_3*100/train_size


def read_waveforms(waveforms, split_time):
    waveforms = [line for line in waveforms if line != '\n']
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        waveforms_total.append(torch.as_tensor([float(elem) for elem in waveforms_list[:(split_time+5)*60]], dtype=torch.float))
    return


def normalize_data(waveforms_tensor, labels_tensor):
    temp_tensor = torch.zeros(waveforms_tensor.size())
    temp_tensor[:, :, 1:] = torch.clone(waveforms_tensor[:, :, :-1])
    res_tensor = waveforms_tensor - temp_tensor

    min_value = torch.min(res_tensor[:, :, 0])
    max_value = torch.max(res_tensor[:, :, 0])
    res_tensor[:, :, 0] -= min_value
    res_tensor[:, :, 0] /= (max_value - min_value)
    # removing 8, 9 categories for lacking data
    new_res_tensor = res_tensor[labels_tensor.round() - 1 < 7].clone()
    return new_res_tensor


def normalize_labels(labels_tensor):
    labels_tensor = torch.round(labels_tensor).long()
    labels_tensor -= 1
    # removing 8, 9 categories for lacking data
    new_labels_tensor = labels_tensor[labels_tensor < 7].clone()
    return new_labels_tensor


def main():
    if KNN is True:
        # path of data while picking waveforms using the knn method
        dir_path = os.path.join(os.getcwd(), f"knn_pick_{n}")
        # path of waveforms tensors
        data_path = f"total_waveforms_tensor_knn_{n}.pt"
        # path of labels tensor
        labels_path = f"total_labels_tensor_knn_{n}.pt"
    if MEAN_VARIANCE is True:
        # path of data while picking waveforms using the mean variance method
        dir_path = os.path.join(os.getcwd(), f"mean_variance_pick_{n}")
        # path of waveforms tensors
        data_path = f"total_waveforms_tensor_mean_variance_{n}.pt"
        # path of labels tensor
        labels_path = f"total_labels_tensor_mean_variance_{n}.pt"
    if VARIANCE is True:
        # path of data while picking waveforms using the variance method
        dir_path = os.path.join(os.getcwd(), f"variance_pick_{n}")
        # path of waveforms tensors
        data_path = f"total_waveforms_tensor_variance_{n}.pt"
        # path of labels tensor
        labels_path = f"total_labels_tensor_variance_{n}.pt"
    if DM_CHOOSE is True:
        # path of data while picking waveforms using the dm method
        dir_path = os.path.join(os.getcwd(), f"dm_pick_{n}")
        # path of waveforms tensors
        data_path = f"total_waveforms_tensor_dm_{n}.pt"
        # path of labels tensor
        labels_path = f"total_labels_tensor_dm_{n}.pt"

    if not os.path.exists(labels_path):
        labels = torch.load("total_labels")
        input_files = np.array(os.listdir(dir_path))
        indices = np.array([int(re.findall('[0-9]+', file)[0]) for file in input_files])
        labels_tensor = labels[indices]
        torch.save(labels_tensor, labels_path)
    else:
        labels_tensor = torch.load(labels_path)

    if not os.path.exists(data_path):
        input_files = np.array(os.listdir(dir_path))
        file_path = [os.path.join(dir_path, input_file) for input_file in input_files]
        input_len = (split_time + 5) * 60
        for file in file_path:
            fd = open(file, 'r')
            lines = fd.readlines()
            read_waveforms(lines, split_time)
            fd.close()
        waveforms_tensor = [F.pad(item, (
        int(np.floor((input_len - item.size()[0]) / 2)), int(np.ceil((input_len - item.size()[0]) / 2))))
                            for item in waveforms_total]
        waveforms_tensor = torch.reshape(torch.stack(waveforms_tensor, 0), (-1, n, input_len))
        torch.save(waveforms_tensor, data_path)
    else:
        waveforms_tensor = torch.load(data_path)

    normalized_waveforms_tensor = normalize_data(waveforms_tensor, labels_tensor)
    normalized_labels_tensor = normalize_labels(labels_tensor)

    x_train, x_test, y_train, y_test = train_test_split(normalized_waveforms_tensor, normalized_labels_tensor,
                                                        test_size=0.2, train_size=0.8, random_state=1, shuffle=True)
    up_samples_ratio = {
        2: 3050,
        5: 2130,
        6: 650
    }

    up_sample = SMOTE(sampling_strategy=up_samples_ratio, random_state=1, k_neighbors=5)

    x_train = x_train.reshape(-1, n * 900)
    x_train, y_train = up_sample.fit_resample(x_train, y_train)
    x_train = x_train.reshape(-1, n, 900)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.FloatTensor(x_test)

    train_size = x_train.size()[0]
    test_size = x_test.size()[0]

    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)

    batch_size = 64
    epochs = 50
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels_arr = y_train.numpy()
    weight = class_weight.compute_class_weight('balanced', np.unique(labels_arr), labels_arr).astype(np.float32)
    print(weight)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weight))
    model = SeismicNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = train_step(model, trainloader, criterion, optimizer, epoch, train_size)
        _, acc_1, acc_2, acc_3 = train_check(model, trainloader, criterion, epoch, train_size)
        test_loss, accuracy_1, accuracy_2, accuracy_3 = test_step(model, testloader, criterion, epoch, test_size)
        print(
            '\n[%d] Train loss: %.3f Test loss: %.3f top 1: %.3f %s top 2: %.3f %s top 3: %.3f %s' % (epoch, train_loss,
                                                                                                      test_loss,
                                                                                                      accuracy_1,
                                                                                                      '%', accuracy_2,
                                                                                                      '%',
                                                                                                      accuracy_3, '%'))
        print(f"acc train : {acc_1}, top 2: {acc_2}, top 3: {acc_3}")

    print('Finished Training')


if __name__ == "__main__":
    main()
