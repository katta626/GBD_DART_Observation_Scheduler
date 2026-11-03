"""
# Local siderial time calculator
# https://stackoverflow.com/questions/55879700/astropy-get-sidereal-time-of-current-location
# https://docs.astropy.org/en/stable/time/index.html
# Arul on 09_10_2025
"""

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import datetime, time
import numpy as np

#GBD observatory

longitude = 77.437547;  # degrees
latitude = 13.603839;  # degrees
height = 713;  # meters

observing_location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)

def get_all_time(): 
    IST = str(np.datetime64(datetime.datetime.now()))
    UTC = datetime.datetime.now(datetime.UTC)
    Astro_fmt = Time(UTC, scale='utc')
    Julian_day = Astro_fmt.jd
    Modified_Julian_day = Astro_fmt.mjd
    
    observing_time = Time(UTC, scale='utc', location=observing_location)
    LST = observing_time.sidereal_time('mean')
    LST = LST.to_string(unit=u.hour, sep=':')

    return IST, UTC, LST, Julian_day, Modified_Julian_day

while True: 
    IST, UTC, LST, JD, MJD = get_all_time()
    print(IST, UTC, LST, JD, MJD)
    time.sleep(1)
