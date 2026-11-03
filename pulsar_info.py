from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, Angle
import datetime, time
import numpy as np

#GBD observatory

longitude = 77.437547;  # degrees
latitude = 13.603839;  # degrees
height = 713;  # meters

observing_location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)

# The scheduler calls RA() for every pulsar once per second; downloading the ATNF
# catalogue each time is slow and hammers the ATNF server, so keep it for a day.
CATALOGUE_MAX_AGE_SECONDS = 24 * 3600
_catalogue = None
_catalogue_loaded_at = 0.0


def get_pulsar_catalogue():
    global _catalogue, _catalogue_loaded_at
    if _catalogue is None or time.time() - _catalogue_loaded_at > CATALOGUE_MAX_AGE_SECONDS:
        from psrqpy import QueryATNF
        _catalogue = QueryATNF(params=['F0']).get_pulsars()
        _catalogue_loaded_at = time.time()
    return _catalogue

def get_all_time():
    IST = str(np.datetime64(datetime.datetime.now()))
    UTC = datetime.datetime.now(datetime.UTC)

    observing_time = Time(UTC, scale='utc', location=observing_location)
    LST = observing_time.sidereal_time('mean')
    #LST.to_string(unit=u.hour, sep=':')
    
    return LST, IST

def RA(k):
    psrs = get_pulsar_catalogue()
    # Get Local Sidereal Time
    lst_now, ist_now = get_all_time()

    #print(lst_now)

    # Example: Target LST from somewhere (maybe from schedule)
    target_lst_str = psrs[k].RAJ  # example: 2:15 AM LST
    
    #target_lst_2=datetime.datetime.strptime(target_lst_str, '%H:%M:%S.%fffffff')
    
    target_lst = Angle(target_lst_str, unit='hour')

    total_seconds = target_lst.hour * 3600
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    today = datetime.date.today()

    ra_datetime = datetime.datetime(
        today.year, today.month, today.day,
        hours, minutes, int(seconds),
        int((seconds % 1)*1e6)
    )

    # Compute difference between LSTs (handle wraparound correctly)
    lst_diff = (target_lst - lst_now).wrap_at(24 * lst_now.unit)

    #print(f"Difference: {lst_diff}")
    #print(f"Difference in hours: {lst_diff.hour:.2f} hrs")

    # Add the LST difference to current IST time to estimate target IST time
    # Convert LST difference in hours to seconds
    seconds_to_wait = lst_diff.hour * 3600

    # Current IST as datetime object
    current_ist_dt = datetime.datetime.now()

    # Future IST time when target LST occurs
    target_ist_dt = current_ist_dt + datetime.timedelta(seconds=seconds_to_wait)

    #print(f"Estimated IST when LST reaches {target_lst_str}: {target_ist_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    return ra_datetime, seconds_to_wait


def count(RA):
    target_lst_str = RA
    lst_now, ist_now = get_all_time()
    target_lst = Angle(target_lst_str, unit='hour')
    lst_diff = (target_lst - lst_now).wrap_at(24 * lst_now.unit)
    
    seconds_to_wait = lst_diff.hour * 3600
    
    return seconds_to_wait
