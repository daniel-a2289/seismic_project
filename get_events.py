from obspy.core.event import read_events
import os
from obspy.clients.fdsn import Client as FDSN_Client


client = FDSN_Client("IRIS")
stations = None
file_path = os.path.join(os.getcwd(), 'events.txt')
events = read_events(file_path, format="QUAKEML")
origin_arr = [event.origins[0] for event in events]

for i, origin in enumerate(origin_arr):
    try:
        client.get_stations(latitude=origin.latitude,
                            longitude=origin.longitude,
                            maxradius=1,  # could be too small, resulting in no waveform received
                            channel="HHZ",
                            level="channel",
                            starttime=origin.time - 300,
                            endtime=origin.time + 600,
                            filename=f"stations/station_{str(i)}.txt")  # may contain a number of waveforms
    except Exception as e:
        print(e)

