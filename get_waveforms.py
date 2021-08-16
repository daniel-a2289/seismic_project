import os
import re

from obspy import read_inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import read_events

FILE_NAME = "station"

client = FDSN_Client("IRIS")
stations_file_path = os.path.join(os.getcwd(), 'stations0')
stations_files = os.listdir(stations_file_path)

events_file_path = os.path.join(os.getcwd(), 'events.txt')
events = read_events(events_file_path, format="QUAKEML")
origin_arr = [event.origins[0] for event in events]

for j in range(1):
    waveforms = None
    labels = []
    event_index = []
    for i, file in enumerate(stations_files[j:j+1]):
        if FILE_NAME in file:
            index = int(re.findall('[0-9]+', file)[0])
            mag_type = events[index].magnitudes[0].magnitude_type
            if mag_type == "Ml" or mag_type == "ML":
                inventory = read_inventory(os.path.join(stations_file_path, file), format="STATIONXML")
                for network in inventory:
                    for station in network:
                        try:
                            new_waveform = client.get_waveforms(network.code, station.code, "*", "HHZ",
                                                                origin_arr[index].time - 300, origin_arr[index].time + 600)
                            if waveforms is None:
                                waveforms = new_waveform
                            else:
                                waveforms.extend(new_waveform)

                            [event_index.append(index) for cnt in range(len(new_waveform))]
                            [labels.append(events[index].magnitudes[0].mag) for cnt in range(len(new_waveform))]
                        except Exception as e:
                            print(e)

    waveforms_file = open(f"waveforms_file_{j}.txt", "w")
    labels_file = open(f"labels_file_{j}.txt", "w")
    event_index_file = open(f"event_index_file_{j}.txt", "w")

    waveforms_file.writelines(waveforms)
    labels_file.writelines(labels)
    event_index_file.writelines(event_index)

    waveforms_file.close()
    labels_file.close()
    event_index_file.close()
