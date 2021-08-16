import os
import re
from obspy import read
import concurrent.futures
import numpy as np

FILE_NAME = "waveforms_"

dir_path = os.path.join(os.getcwd(), "results")
files = os.listdir(dir_path)


def get_data(file):
    if file.startswith(FILE_NAME):
        waveforms = []
        file_path = os.path.join(os.getcwd(), dir_path, file)
        data_path = os.listdir(file_path)
        for waveform in data_path:
            path = os.path.join(file_path, waveform)
            new_waveform_data = read(path, format='MSEED')[0].data
            waveforms.append(list(new_waveform_data))

        index = int(re.findall('[0-9]+', file)[0])
        fd = open(f"data_{index}.txt", 'w+')
        for items in waveforms:
            fd.write('%s\n' % items)
        fd.close()


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(get_data, files)


