import logging
import time
import json
import urllib

from mwa_source_finder import logger_setup


def get_metadata(
    servicetype: str = "metadata",
    service: str = "obs",
    params: dict = None,
    retries: int = 3,
    retry_http_error: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """Function to call a JSON web service to perform an MWA metadata call.

    Parameters
    ----------
    servicetype : str, optional
        Either the 'observation' which makes human readable html pages or
        'metadata' which returns data, by default 'metadata'.
    service : str, optional
        The meta data service out of ['obs', 'find', 'con'], by default 'obs'.
            obs: Returns details about a single observation.
            find: Search the database for observations that satisfy given criteria.
            con: Finds the configuration information for an observation.
    params : dict, optional
        A dictionary of the options to use in the metadata call which is dependent
        on the service, by default None.
    retries : int, optional
        The number of times to retry timeout errors, by default 3.
    retry_http_error : bool, optional
        Whether to retry the request after a HTTP error, by default False.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    dict
        The result for that service.
    """
    # Append the service name to this base URL, eg 'con', 'obs', etc.
    BASEURL = "http://ws.mwatelescope.org/"

    if params:
        # Turn the dictionary into a string with encoded 'name=value' pairs
        data = urllib.parse.urlencode(params)
    else:
        data = ""

    # Try several times (3 by default)
    wait_time = 30
    result = None
    for _ in range(0, retries):
        err = False
        try:
            result = json.load(
                urllib.request.urlopen(
                    BASEURL + servicetype + "/" + service + "?" + data
                )
            )
        except urllib.error.HTTPError as err:
            logger.error(
                f"HTTP error from server: code={err.code}, response: {err.read()}"
            )
            if retry_http_error:
                logger.error(f"Waiting {wait_time} seconds and trying again")
                time.sleep(wait_time)
                pass
            else:
                logger.error(err)
                break
        except urllib.error.URLError as err:
            logger.error(f"URL or network error: {err.reason}")
            logger.error(f"Waiting {wait_time} seconds and trying again")
            time.sleep(wait_time)
            pass
        else:
            break
    else:
        logger.error(f"Tried {retries} times. Exiting.")

    return result


def get_common_metadata(obsid: int, logger: logging.Logger = None) -> dict:
    """Get observation metadata and extract some commonly used data.

    Parameters
    ----------
    obsid : int
        The observation ID.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    dict
        A dictionary of commonly used metadata:
            duration : int
                The observation duration in seconds.
            delays : list
                A list with two items:
                    xdelays : list
                        The delays for the X polarisation.
                    ydelays : list
                        The delays for the Y polarisation.
            channels : list
                The frequency channels in MHz.
            bandwidth : float
                The bandwidth in MHz.
            centrefreq : float
                The centre frequency in MHz.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    metadata = get_metadata(service="obs", params={"obs_id": obsid}, logger=logger)
    if metadata is None:
        logger.error(f"Could not get metadata for obs ID: {obsid}")
        return None

    try:
        minfreq = float(min(metadata["rfstreams"]["0"]["frequencies"]))
        maxfreq = float(max(metadata["rfstreams"]["0"]["frequencies"]))
        xdelays = metadata["rfstreams"]["0"]["xdelays"]
        ydelays = metadata["rfstreams"]["0"]["ydelays"]
        channels = metadata["rfstreams"]["0"]["frequencies"]
    except KeyError:
        logger.error(f"Incomplete metadata for obs ID: {obsid}")
        return None

    common_metadata = dict(
        duration=metadata["stoptime"] - metadata["starttime"],
        delays=[xdelays, ydelays],
        channels=channels,
        bandwidth=1.28 * (channels[-1] - channels[0] + 1),
        centrefreq=1.28 * (minfreq + 0.5 * (maxfreq - minfreq)),
    )
    return common_metadata


def get_all_obsids(pagesize: int = 50, logger: logging.Logger = None) -> list:
    """Loops over pages for each page for MWA metadata calls.

    Parameters
    ----------
    pagesize : int
        Size of the page to query at a time.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    obsids : list
        A list of the MWA observation IDs.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    legacy_params = {"mode": "VOLTAGE_START"}
    mwax_params = {"mode": "MWAX_VCS"}
    obsids = []
    for params in [legacy_params, mwax_params]:
        logger.debug(f"Searching {params['mode']} observations")
        params["pagesize"] = pagesize
        temp = []
        page = 1
        # Need to ask for a page of results at a time
        while len(temp) == pagesize or page == 1:
            params["page"] = page
            logger.debug(f"Page: {page}  params: {params}")
            temp = get_metadata(
                service="find", params=params, retry_http_error=True, logger=logger
            )
            # If there are no obs in the field (which is rare), None is returned
            if temp is not None:
                for row in temp:
                    obsids.append(row[0])
            else:
                temp = []
            page += 1
    return obsids
