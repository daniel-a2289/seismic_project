# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import absolute_import
# from more_itertools import map_except
import time
# import concurrent.futures
# import numpy as np
# import random
# import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
# from obspy import read_inventory

start = time.perf_counter()
client = FDSN_Client("IRIS")
# results = None

st = UTCDateTime("2014-3-27 11:00:00")
# t1 = UTCDateTime("2001-01-07T 00:00:00")
et = UTCDateTime("2020-11-16 11:00:00")

ADD_TIME = 15780000

date = st
new_date = date + ADD_TIME

count = 0
while date < et:
    try:
        cat = client.get_events(minmagnitude=6, maxmagnitude=10, starttime=date, endtime=new_date)
        cat.write(filename=f"cat610_{str(count)}.txt", format="QUAKEML")
    except Exception as e:
        print(e)
        # exit()
    date = new_date
    new_date = date + ADD_TIME
    count += 1

exit()

# st = "2016-11-13 11:00:00.000"
# et = "2017-11-13 11:00:00.000"

# cat13 = client.get_events(minmagnitude=1, maxmagnitude=3, starttime=st, endtime=et)
# cat34 = client.get_events(minmagnitude=3, maxmagnitude=4, starttime=st, endtime=et)
# cat45 = client.get_events(minmagnitude=4, maxmagnitude=5, starttime=st, endtime=et)
# cat56 = client.get_events(minmagnitude=5, maxmagnitude=6, starttime=st, endtime=et)

# st = "2008-11-13 11:00:00.000"
# et = "2020-11-16 11:00:00.000"

# cat610 = client.get_events(minmagnitude=6, maxmagnitude=10, starttime=st, endtime=et)


# def getwaveforms(network):
#     try:
#       return station_arr = [client.get_waveforms(network.code, station.code, "*", "HHZ",
#                                       otime-300, otime + 600) for station in network]
#
#     except:
#       return None #has none when no waveforms are found

try:
  avg_arr = np.zeros(np.shape(st)[0])
  avg_arr = [np.average(val.data) for val in st] ##show the results with average and with k means

  data_cnt = np.argmax(avg_arr)
  data = st[data_cnt].data
  data_stack.append((data))
except:
  data_stack.append((0))
print(data_stack[-1])

from obspy import Stream
data_stack = []
i = 0
for event in cat:
  i += 1
  if i % 1000 == 0:
    print (100*i/datasize)

  origin = event.origins[0]
  otime = origin.time
  try:
    inventory = client.get_stations(latitude=event.origins[0].latitude,
                                  longitude=event.origins[0].longitude,
                                  maxradius=0.5, # could be too small, resulting in no waveform recieved
                                  channel="HHZ",
                                  level="channel",
                                  starttime = otime-300,
                                  endtime = otime+600)
  except:
    data_stack.append((0))
    pass
  st = Stream()
  for network in inventory:
    for station in network:
        try:
            st += client.get_waveforms(network.code, station.code, "*", "HHZ",
                                      otime-300, otime + 600)
        except:
            pass
  try:
    avg_arr = np.zeros(np.shape(st)[0])
    avg_arr = [np.average(val.data) for val in st]

    data_cnt = np.argmax(avg_arr)
    data = st[data_cnt].data
    data_stack.append((data))
  except:
    data_stack.append((0))
  print(data_stack[-1])

random.shuffle(cat)

train = cat[:int(0.8*datasize)]
test  = cat[int(0.8*datasize):]

'''