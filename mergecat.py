from obspy.core.event import read_events
import os

CAT_TYPE = "Full"

files = os.listdir()
catalog = None
for file in files:
    if CAT_TYPE in file:
        file_path = os.path.join(os.getcwd(), file)
        new_events = read_events(file_path, format="QUAKEML")
        if catalog is None:
            catalog = new_events
        else:
            catalog.extend(new_events)

catalog.write(f"{CAT_TYPE}-Full.txt", format="QUAKEML")
