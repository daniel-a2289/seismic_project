import os
import re
from obspy import read
import concurrent.futures
import numpy as np
# import pydiffmap as pf

VARIANCE = False
MEAN_VARIANCE = False
DM = False
KNN = True
dir_path = os.path.join(os.getcwd(), "data")
files = os.listdir(dir_path)
# number of waveforms taken from each event
n = 1


def variance_choose(waveforms, n):
    """
    finds n waveforms per event with smallest variance
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: n chosen indices of waveforms
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
    finds n waveforms per event which are closest to the mean variance
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: n chosen indices of waveforms
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
    finds n waveforms per event which are closest to the mean waveform
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: n chosen indices of waveforms
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

'''
def dm_choose(waveforms, n):
    """
    applies diffusion maps transformation to lower dimensions,
    and finds n waveforms per event which are closest to the mean waveform
    :param waveforms: waveforms of one specific event
    :param n: the amount of wanted waveforms from each event
    :return: n chosen indices of waveforms
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
    embaded_data = dm(waveforms_mat)
    dm_indx = find_nearest2mean(embaded_data, n)
    return dm_indx


def dm(input_matrix):
    n_evecs = 3
    epsilon = 'bgh'
    alpha = 0.5
    k = 49
    kernel_type = 'gaussian'
    metric = 'euclidean'
    bandwidth_normalize = False
    oos = 'nystroem'
    dim = 3
    my_kernel = pf.kernel.Kernel(kernel_type, epsilon, k, metric=metric)
    my_dmap = pf.diffusion_map.DiffusionMap(my_kernel, alpha, n_evecs, bandwidth_normalize=bandwidth_normalize, oos=oos)
    my_dmap.dmap = my_dmap.fit_transform(input_matrix)
    dmap_embedded_data = my_dmap.dmap
    # print('embedded data:\n{}'.format(dmap_embedded_data))
    # print('size of embedded data:\n{}'.format(np.size(dmap_embedded_data)))
    # values = my_dmap.evals
    # dm.visualization.embedding_plot(my_dmap, dim, show=True)
    return dmap_embedded_data
'''


def choosing(file):
    file_path = os.path.join(dir_path, file)
    fd = open(file_path, 'r')
    lines = fd.readlines()

    if VARIANCE is True:
        indices = variance_choose(lines, n)
        chosen_dir_path = f"variance_pick_{n}"
        data_file_name = "variance_pick_data_"
    if MEAN_VARIANCE is True:
        indices = mean_variance_choose(lines, n)
        chosen_dir_path = f"mean_variance_pick_{n}"
        data_file_name = "mean_variance_pick_data_"
    if DM is True:
        indices = dm_choose(lines, n)
        chosen_dir_path = f"dm_pick_{n}"
        data_file_name = "dm_pick_data_"
    if KNN is True:
        indices = knn_choose(lines, n)
        chosen_dir_path = f"knn_pick_{n}"
        data_file_name = "knn_pick_data_"

    if indices is None:
        fd.close()
        return
    index = int(re.findall('[0-9]+', file)[0])
    file_d = open(os.path.join(chosen_dir_path, str(data_file_name) + f"{index}.txt"), 'w+')
    for item in indices:
        file_d.write('%s\n' % lines[item])
    file_d.close()
    fd.close()
    return


def main():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(choosing, files)


if __name__ == "__main__":
    main()

