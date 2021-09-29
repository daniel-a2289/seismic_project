# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import absolute_import
import time
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client

start = time.perf_counter()
client = FDSN_Client("IRIS")

st = UTCDateTime("2000-1-1 11:00:00")
# earlier start time for magnitudes bigger then 6
# st = UTCDateTime("1971-1-1 11:00:00")
et = UTCDateTime("2008-11-13 11:00:00")


ADD_TIME = 15780000

date = st
new_date = date + ADD_TIME
catalog = None
count = 0
while date < et:
    try:
        cat = client.get_events(minmagnitude=6, maxmagnitude=8, starttime=date, endtime=new_date)
    except Exception as e:
        print(e)
    if catalog is None:
        catalog = cat
    else:
        catalog.extend(cat)
    date = new_date
    new_date = date + ADD_TIME
    count += 1

catalog.write(f"events.txt", format="QUAKEML")
exit()

