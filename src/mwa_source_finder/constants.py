from astropy import units as u
from astropy.coordinates import EarthLocation

__all__ = [
    "TEL_LAT",
    "TEL_LON",
    "TEL_ELEV",
    "TEL_LOCATION",
    "LINE_STYLES",
]

TEL_LAT = -26.703319
TEL_LON = 116.67081
TEL_ELEV = 377.827
TEL_LOCATION = EarthLocation(lat=TEL_LAT * u.deg, lon=TEL_LON * u.deg, height=TEL_ELEV * u.m)
LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1, 1, 1, 1, 1)),
    (0, (5, 1, 1, 1, 1, 1, 1, 1)),
    (0, (5, 1, 5, 1, 1, 1)),
    (0, (5, 1, 5, 1, 1, 1, 1, 1)),
    (0, (5, 1, 5, 1, 5, 1, 1, 1)),
    (0, (5, 1, 5, 1, 5, 1, 1, 1, 1, 1)),
    (0, (5, 1, 1, 1, 5, 1, 5, 1, 1, 1)),
    (0, (5, 1, 1, 1, 5, 1, 1, 1, 1, 1)),
]
