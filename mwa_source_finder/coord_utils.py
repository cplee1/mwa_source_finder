from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u
import psrqpy

from mwa_source_finder import logger_setup


def is_float(string):
    """Check if a given string is a valid float.

    Parameters
    ----------
    string : `str`
        The string to check

    Returns
    -------
    is_float : `bool`
        True if the string is a valid float
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int(string):
    """Check if a given string is a valid integer.

    Parameters
    ----------
    string : `str`
        The string to check

    Returns
    -------
    is_int : `bool`
        True if the string is a valid integer
    """
    if string.isnumeric():
        return True
    else:
        return False


def is_sexigesimal(coord, mode):
    """Check if a given sexigesimal string is valid.

    Parameters
    ----------
    coord : `str`
        The sexigesimal coordinate to check
    mode : `str`
        If 'RA', use hours, else if 'DEC', use degrees

    Returns
    -------
    is_sexigesimal : `bool`
        True if the string is a valid sexigesimal
    """
    if mode == "RA":
        hours, minutes, seconds = coord.split(":")
        if not is_int(hours):
            return False
        if int(hours) < 0 or int(hours) > 24:
            return False
    elif mode == "DEC":
        degrees, minutes, seconds = coord.split(":")
        if degrees.startswith(("-", "–", "+")):
            degrees = degrees[1:]
        if not is_int(degrees):
            return False
        if int(degrees) < 0 or int(degrees) > 90:
            return False
    if not is_int(minutes):
        return False
    if int(minutes) < 0 or int(minutes) > 60:
        return False
    if not is_float(seconds):
        return False
    if float(seconds) < 0.0 or float(seconds) > 60.0:
        return False
    return True


def format_sexigesimal(coord, add_sign=False, logger=None):
    """Format a sexigesimal coordinate properly. Will assume zero for any
    missing units. E.g. '09:23' will be formatted as '09:23:00.00'.

    Parameters
    ----------
    coord : `str`
        A sexigesimal coordinate
    add_sign : `bool`, optional
        If True, will add a sign to the output (Default: False)
    logger : `logging.Logger`, optional
        A custom logger to use

    Returns
    -------
    formatted_coord : `str`
        The properly formatted coordinate
    """
    if logger is None:
        logger = logger_setup.get_logger()

    # Determine the sign
    if coord.startswith(("-", "–")):
        sign = "-"
        coord = coord[1:]
    elif coord.startswith("+"):
        sign = "+"
        coord = coord[1:]
    else:
        sign = "+"
    # Split up the units
    parts = coord.split(":")
    # Fill in the missing units
    if len(parts) == 3:
        deg = int(parts[0])
        minute = int(parts[1])
        second = float(parts[2])
    elif len(parts) == 2:
        deg = int(parts[0])
        minute = int(float(parts[1]))
        second = 0.0
    elif len(parts) == 1:
        deg = int(float(parts[0]))
        minute = 0
        second = 0.0
    else:
        logger.error(f"Cannot interpret coordinate: {coord}.")
        return None
    # Add the sign back if specified
    if add_sign:
        formatted_coord = f"{sign}{int(deg):02d}:{int(minute):02d}:{second:05.2f}"
    else:
        formatted_coord = f"{int(deg):02d}:{int(minute):02d}:{second:05.2f}"
    return formatted_coord


def decimal_to_sexigesimal(rajd, decjd):
    """Convert decimal degrees into sexagesimal coordinates.

    Parameters
    ----------
    rajd : `float`
        The right acension in decimal degrees
    decjd : `float`
        The declination in decimal degrees

    Returns
    -------
    raj : `str`
        The right acension in sexigesimal format ("HH:MM:SS.SSSS")
    decj : `str`
        The declination in sexigesimal format ("DD:MM:SS.SSSS")
    """
    c = SkyCoord(rajd, decjd, frame="icrs", unit=(u.deg, u.deg))
    raj = c.ra.to_string(unit=u.hour, sep=":")
    decj = c.dec.to_string(unit=u.degree, sep=":")
    return raj, decj


def sexigesimal_to_decimal(raj, decj):
    """Convert sexagesimal coordinates to decimal degrees.

    Parameters
    ----------
    raj : `str`
        The right acension in sexigesimal format ("HH:MM:SS.SSSS")
    decj : `str`
        The declination in sexigesimal format ("DD:MM:SS.SSSS")

    Returns
    -------
    rajd : `float`
        The right acension in decimal degrees
    decjd : `float`
        The declination in decimal degrees
    """
    c = SkyCoord(raj, decj, frame="icrs", unit=(u.hourangle, u.deg))
    rajd = c.ra.deg
    decjd = c.dec.deg
    return rajd, decjd


def equatorial_to_horizontal(rajd, decjd, gps_epoch):
    """Convert equatorial (RA/DEC) to horizontal (Alt/Az) coordinates.

    Parameters
    ----------
    rajd : `float` or `numpy.array`
        The right acension in decimal degrees
    decjd : `float` or `numpy.array`
        The declination in decimal degrees
    gps_epoch : `float` or `numpy.array`
        The gps time of the horizontal coordinates

    Returns
    -------
    alt : `float` or `numpy.array`
        The altitude angle in degrees
    az : `float` or `numpy.array`
        The azimuth angle in degrees
    za : `float` or `numpy.array`
        The zenith angle in degrees
    """
    eq_pos = SkyCoord(rajd, decjd, unit=(u.deg, u.deg))
    obstime = Time(float(gps_epoch), format="gps")
    earth_location = EarthLocation.from_geodetic(
        lon="116:40:14.93", lat="-26:42:11.95", height=377.8
    )
    altaz_pos = eq_pos.transform_to(AltAz(obstime=obstime, location=earth_location))
    alt = altaz_pos.alt.deg
    az = altaz_pos.az.deg
    za = 90.0 - alt
    return alt, az, za


def get_pulsar_coords(pulsar, query, logger=None):
    """Get pulsar coordinates from a psrqpy query.

    Parameters
    ----------
    pulsar : `str`
        Pulsar Jname or Bname
    query : `query`
        An instance of the psrqpy.QueryATNF class
    logger : `logging.Logger`, optional
        A custom logger to use

    Returns
    -------
    raj : `str`
        J2000 right ascension in sexigesimal format
    decj : `str`
        J2000 declination in sexigesimal format
    rajd : `str`
        J2000 right ascension in decimal degrees format
    decjd : `str`
        J2000 declination in decimal degrees format
    """
    if logger is None:
        logger = logger_setup.get_logger()

    # If pulsar Bname, convert to Jname
    if pulsar.startswith("B"):
        try:
            pid = list(query["PSRB"]).index(pulsar)
            pulsar = query["PSRJ"][pid]
        except ValueError:
            logger.error(f"Pulsar not found in catalogue: {pulsar}")
            return None, None, None, None
    # Check if pulsar is in the catalogue
    if pulsar not in list(query["PSRJ"]):
        logger.error(f"Pulsar not found in catalogue: {pulsar}")
        return None, None, None, None
    # Get coordinates from the query
    psrs = query.get_pulsars()
    raj = psrs[pulsar].RAJ
    decj = psrs[pulsar].DECJ
    rajd = psrs[pulsar].rajd
    decjd = psrs[pulsar].DECJD
    return raj, decj, rajd, decjd


def interpret_coords(coords, logger=None):
    """Interpret source coordinates.

    Parameters
    ----------
    coords : `str`
        Coordinates in <RA>_<DEC> format, either sexigesimal or decimal
    logger : `logging.Logger`, optional
        A custom logger to use

    Returns
    -------
    raj : `str`
        J2000 right ascension in sexigesimal format
    decj : `str`
        J2000 declination in sexigesimal format
    rajd : `str`
        J2000 right ascension in decimal degrees format
    decjd : `str`
        J2000 declination in decimal degrees format
    """
    if logger is None:
        logger = logger_setup.get_logger()

    # Split up the coordinates
    raj = coords.split("_")[0]
    decj = coords.split("_")[1]
    # Check how the coordinates are formatted
    if ":" in raj or ":" in decj:
        if ":" not in raj or ":" not in decj:
            logger.error(f"Inconsistent coordinate formats: {coords}")
            return None, None, None, None
        # Must be a sexigesimal
        decimal_flag = False
        # Add positive sign if no sign is present
        if decj[0].isdigit():
            decj = f"+{decj}"
    elif is_float(raj) and is_float(decj):
        # Must be a decimal
        decimal_flag = True
    else:
        logger.error(f"Coordinates not valid: {coords}")
        return None, None, None, None
    if decimal_flag:
        # Convert coordinates to sexigesimal
        rajd = raj
        decjd = decj
        raj, decj = decimal_to_sexigesimal(rajd, decjd)
    else:
        # Check that the coordinates are valid
        if not is_sexigesimal(raj, "RA") or not is_sexigesimal(decj, "DEC"):
            logger.error(f"Invalid sexigesimal format: {coords}")
            return None, None, None, None
        rajd, decjd = sexigesimal_to_decimal(raj, decj)
    # Make sure sexigesimal coords are formatted correctly
    raj = format_sexigesimal(raj, logger=logger)
    decj = format_sexigesimal(decj, add_sign=True, logger=logger)
    return raj, decj, rajd, decjd


def get_pointings(sources, logger=None):
    """Get source pointing information and store it in dictionary format.

    Parameters
    ----------
    sources : `list`
        A list of source names
    logger : `logging.Logger`, optional
        A custom logger to use

    Returns
    -------
    pointings : `list`
        A list of dictionaries with the following items: Name, RAJ, DEC, RAJD, DECJD
    """
    if logger is None:
        logger = logger_setup.get_logger()

    pointings = []
    query_flag = False
    # Check for pulsar names
    for source in sources:
        if source.startswith(("J", "B")):
            query_flag = True
            break
    # If required, query the catalogue
    if query_flag:
        logger.info("Querying the pulsar catalogue")
        query = psrqpy.QueryATNF(
            params=["PSRJ", "PSRB", "RAJ", "DECJ", "RAJD", "DECJD"]
        )
        logger.info(f"Using ATNF pulsar catalogue version {query.get_version}")
    # Loop through all sources, get pointings and add them to dictionaries
    for source in sources:
        if source.startswith(("J", "B")):
            logger.debug(f'Treating "{source}" as a pulsar')
            raj, decj, rajd, decjd = get_pulsar_coords(source, query, logger=logger)
        elif "_" in source:
            logger.debug(f'Treating "{source}" as coordinates')
            raj, decj, rajd, decjd = interpret_coords(source, logger=logger)
        else:
            logger.error(f"Source not recognised: {source}")
        if raj is None:
            continue
        pointing = dict(Name=source, RAJ=raj, DECJ=decj, RAJD=rajd, DECJD=decjd)
        pointings.append(pointing)
    return pointings


def get_atnf_pulsars(logger=None):
    """Get source pointing information from the ATNF pulsar catalogue.

    Parameters
    ----------
    logger : `logging.Logger`, optional
        A custom logger to use

    Returns
    -------
    pointings : `list`
        A list of dictionaries with the following items: Name, RAJ, DEC, RAJD, DECJD
    """
    if logger is None:
        logger = logger_setup.get_logger()

    logger.info("Querying the pulsar catalogue...")
    query = psrqpy.QueryATNF(params=["PSRJ", "RAJ", "DECJ", "RAJD", "DECJD"])
    logger.info(f"Using ATNF pulsar catalogue version {query.get_version}")
    psrjs = list(query.table["PSRJ"])
    rajs = list(query.table["RAJ"])
    decjs = list(query.table["DECJ"])
    rajds = list(query.table["RAJD"])
    decjds = list(query.table["DECJD"])
    # Loop through all the pulsars and store the pointings in dictionaries
    pointings = []
    for psrj, raj, decj, rajd, decjd in zip(psrjs, rajs, decjs, rajds, decjds):
        raj = format_sexigesimal(raj, logger=logger)
        decj = format_sexigesimal(decj, add_sign=True, logger=logger)
        pointing = dict(Name=psrj, RAJ=raj, DECJ=decj, RAJD=rajd, DECJD=decjd)
        pointings.append(pointing)
    return pointings
