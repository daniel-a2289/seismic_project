import os
import re
from obspy import read
import concurrent.futures
import warnings
import numpy as np
from obspy.io.mseed import InternalMSEEDWarning

FILE_NAME = "waveforms_"

warnings.simplefilter("error", InternalMSEEDWarning)
dir_path = os.path.join(os.getcwd(), "results")
output_path = os.path.join(os.getcwd(), "data")
files = os.listdir(dir_path)


def get_data(file):
    if file.startswith(FILE_NAME):
        waveforms = []
        file_path = os.path.join(os.getcwd(), dir_path, file)
        data_path = os.listdir(file_path)
        for waveform in data_path:
            path = os.path.join(file_path, waveform)
            try:
                new_waveform_data = read(path, format='MSEED')[0].data
            except InternalMSEEDWarning as e:
                print(f"the waveform {file} is corrupted: " + str(e))
                continue
            waveforms.append(list(new_waveform_data))

        # true if the list is not empty
        if waveforms:
            index = int(re.findall('[0-9]+', file)[0])
            fd = open(os.path.join(output_path, f"data_{index}.txt"), 'w+')
            for items in waveforms:
                fd.write('%s\n' % items)
            fd.close()


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(get_data, files)


