
import os
import obspy
from obspy.core.event import read_events
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

j = 0

# do all the next stages for every single event in events array

# for i, event in enumerate(events[5375:]):


def download(i, event):
    origin_time = event.origins[0].time
    origin_latitude = event.origins[0].latitude
    origin_longitude = event.origins[0].longitude

    # Circular domain around the epicenter. This will download all data between
    # 70 and 90 degrees distance from the epicenter. This module also offers
    # rectangular and global domains. More complex domains can be defined by
    # inheriting from the Domain class.
    domain = CircularDomain(latitude=origin_latitude, longitude=origin_longitude,
                            minradius=90, maxradius=91)

    # domain = RectangularDomain(minlatitude=origin_latitude - 10, maxlatitude=origin_latitude + 10,
    #                            minlongitude=origin_longitude - 10, maxlongitude=origin_longitude + 10)

    restrictions = Restrictions(
        # Get data from 5 minutes before the event to one hour after the
        # event. This defines the temporal bounds of the waveform data.
        starttime=origin_time - 5 * 60,
        endtime=origin_time + 3600,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=True,
        # And you might only want waveforms that have data for at least 95 % of
        # the requested time span. Any trace that is shorter than 95 % of the
        # desired total duration will be discarded.
        minimum_length=0.95,
        # No two stations should be closer than 10 km to each other. This is
        # useful to for example filter out stations that are part of different
        # networks but at the same physical station. Settings this option to
        # zero or None will disable that filtering.
        minimum_interstation_distance_in_m=10E3,
        # Only HH or BH channels. If a station has HH channels, those will be
        # downloaded, otherwise the BH. Nothing will be downloaded if it has
        # neither. You can add more/less patterns if you like.
        channel_priorities=["LHZ"],
        # Location codes are arbitrary and there is no rule as to which
        # location is best. Same logic as for the previous setting.
        location_priorities=["", "01", "00", "EP", "S1", "S3", "02", "10", "09", "08",
                             "03", "04", "06", "07", "05", "20", "T0", "2C", "40", "50"])

    # No specified providers will result in all known ones being queried.
    mdl = MassDownloader(providers=["IRIS"])
    # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
    # folders with automatically chosen file names.
    mdl.download(domain, restrictions, mseed_storage=f"waveforms_{i+j}", stationxml_storage=f"stations_{i+j}")


if __name__ == "__main__":
    events_file_path = os.path.join(os.getcwd(), 'events.txt')
    events = read_events(events_file_path, format="QUAKEML")

    print("finished reading events")

    with Pool(cpu_count()) as pool:
        values = [(i, event) for i, event in enumerate(events[j:])]
        # print(values)
        values = tqdm(values, total=len(values))
        pool.starmap(download, values)
