import os
import re
from obspy import read
import concurrent.futures
import numpy as np

VARIANCE = True
MEAN_VARIANCE = False
DM_CHOOSE = False
KNN = False
dir_path = os.path.join(os.getcwd(), "data")
files = os.listdir(dir_path)
n = 3


def variance_choose(waveforms, n):
    """
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: chosen indices of waveforms with n smallest variance
    """
    length = len(waveforms)
    if length == n:
        return np.arange(length)
    if length < n:
        return None
    variance = np.zeros(length)
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        waveforms_arr = np.array([int(elem) for elem in waveforms_list])
        variance[i] = np.var(waveforms_arr)
    index = np.argpartition(variance, n)[:n]  # n waveforms with smallest variance
    return index


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def mean_variance_choose(waveforms, n):
    """
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: chosen indices of waveforms with n closest to mean variance
    """
    length = len(waveforms)
    if length == n:
        return np.arange(length)
    if length < n:
        return None
    variance = np.zeros(length)
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        waveforms_arr = np.array([int(elem) for elem in waveforms_list])
        variance[i] = np.var(waveforms_arr)

    mean_var = np.average(variance)
    index = []
    for j in range(n):
        ind = find_nearest(variance, mean_var)
        index.append(ind)
        variance[ind] = np.inf

    return np.array(index)


def find_nearest2mean(mean, waveforms_mat, n):
    dist_array = np.linalg.norm(waveforms_mat - mean, axis=1)
    index = np.argpartition(dist_array, n)[:n]
    return index


def knn_choose(waveforms, n):
    """
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: chosen indices of waveforms with n nearest neighbors
    """
    length = len(waveforms)
    if length == n:
        return np.arange(length)
    if length < n:
        return None
    for i, item in enumerate(waveforms):
        waveforms_list = item[1:-2].split(", ")  # to remove brackets
        if i == 0:
            waveforms_mat = np.zeros((length, len(waveforms_list)))
        waveforms_mat[i, :] = np.array([int(elem) for elem in waveforms_list])
    total = np.sum(waveforms_mat, axis=0)
    mean = total/length
    index = find_nearest2mean(mean, waveforms_mat, n)
    return index


def choosing(file):
    file_path = os.path.join(dir_path, file)
    fd = open(file_path, 'r')
    lines = fd.readlines()

    if VARIANCE is True:
        indices = variance_choose(lines, n)
    if MEAN_VARIANCE is True:
        indices = mean_variance_choose(lines, n)
    if DM_CHOOSE is True:
        indices = variance_choose(lines, n)
    if KNN is True:
        indices = knn_choose(lines, n)

    if indices is None:
        fd.close()
        return
    index = int(re.findall('[0-9]+', file)[0])
    file_d = open(f"variance_pick_data_{index}.txt", 'w+')
    for item in indices:
        file_d.write('%s\n' % lines[item])
    file_d.close()
    fd.close()
    return


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(choosing, files)
