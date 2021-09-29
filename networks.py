from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from initial_nn import normalize_data, normalize_labels
import torch
from sklearn.model_selection import train_test_split
import numpy as np

n = 1
DATA_PATH = f"total_waveforms_tensor_mean_variance_{n}.pt"
LABELS_PATH = f"total_labels_tensor_mean_variance_{n}.pt"

labels_tensor = torch.load(LABELS_PATH)
waveforms_tensor = torch.load(DATA_PATH)

normalized_waveforms_tensor = normalize_data(waveforms_tensor, labels_tensor)
normalized_labels_tensor = normalize_labels(labels_tensor)

x_train, x_test, y_train, y_test = train_test_split(normalized_waveforms_tensor, normalized_labels_tensor,
                                                    test_size=0.2, train_size=0.8, random_state=1, shuffle=True)

x_train = torch.FloatTensor(x_train)
x_train = torch.flatten(x_train, start_dim=1)
x_test = torch.FloatTensor(x_test)
x_test = torch.flatten(x_test, start_dim=1)
acc_1, acc_2, acc_3 = [], [], []

for j in range(3, 30):
    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(x_train, y_train)
    pred_j = knn.predict_proba(x_test)
    pred_j = torch.from_numpy(pred_j)

    temp = [pred_j[i].argmax() == y_test[i] for i in range(len(y_test))]
    acc_1.append(np.sum(temp)*100/x_test.size()[0])

    temp_1 = [any(torch.argsort(pred_j[i])[-2:] == y_test[i]) for i in range(len(y_test))]
    acc_2.append(np.sum(temp_1)*100/x_test.size()[0])

    temp_0 = [any(torch.argsort(pred_j[i])[-3:] == y_test[i]) for i in range(len(y_test))]
    acc_3.append(np.sum(temp_0)*100/x_test.size()[0])

    print(f"for {j} neighbors: top 1 : {acc_1[-1]}, top 2: {acc_2[-1]}, top 3: {acc_3[-1]}")




