#!/usr/bin/env python

import time
import json
import urllib

from logger_module import get_logger


def getmeta(servicetype='metadata', service='obs', params=None, retries=3, retry_http_error=False, logger=None):
    """Function to call a JSON web service to perform an MWA metadata call.
    Taken verbatim from http://mwa-lfd.haystack.mit.edu/twiki/bin/view/Main/MetaDataWeb.

    Parameters
    ----------
    servicetype : `str`
        Either the 'observation' which makes human readable html pages or
        'metadata' which returns data (Default: 'metadata')
    service : `str`
        The meta data service out of ['obs', 'find', 'con'] (Default: 'obs')
            obs: Returns details about a single observation
            find: Search the database for observations that satisfy given criteria
            con: Finds the configuration information for an observation
    params : `dict`
        A dictionary of the options to use in the metadata call which is dependent on the service
    retries : `int`, optional
        The number of times to retry timeout errors (Default: 3)

    Returns
    -------
    result : `dict`
        The result for that service
    """
    # Append the service name to this base URL, eg 'con', 'obs', etc.
    BASEURL = 'http://ws.mwatelescope.org/'

    if params:
        # Turn the dictionary into a string with encoded 'name=value' pairs
        data = urllib.parse.urlencode(params)
    else:
        data = ''
    
    # Try several times (3 by default)
    wait_time = 30
    result = None
    for x in range(0, retries):
        err = False
        try:
            result = json.load(urllib.request.urlopen(BASEURL + servicetype + '/' + service + '?' + data))
        except urllib.error.HTTPError as err:
            logger.error("HTTP error from server: code=%d, response: %s" % (err.code, err.read()))
            if retry_http_error:
                logger.error("Waiting {} seconds and trying again".format(wait_time))
                time.sleep(wait_time)
                pass
            else:
                raise err
                break
        except urllib.error.URLError as err:
            logger.error("URL or network error: %s" % err.reason)
            logger.error("Waiting {} seconds and trying again".format(wait_time))
            time.sleep(wait_time)
            pass
        else:
            break
    else:
        logger.error("Tried {} times. Exiting.".format(retries))

    return result


def get_all_obsids(pagesize=50, logger=None):
    """Loops over pages for each page for MWA metadata calls

    Parameters
    ----------
    params : `dict`
        The dictionary of constraints used to search for suitable observations
        (Default: {'mode':'VOLTAGE_START'})

    Returns
    -------
    obsids : `list`
        List of the MWA observation IDs
    """
    if logger is None:
        logger = get_logger()

    legacy_params = {'mode':'VOLTAGE_START'}
    mwax_params = {'mode':'MWAX_VCS'}
    for params in [legacy_params, mwax_params]:
        logger.debug(f"Searching {params['mode']} observations")
        params['pagesize'] = pagesize
        obsids = []
        temp = []
        page = 1
        # Need to ask for a page of results at a time
        while len(temp) == pagesize or page == 1:
            params['page'] = page
            logger.debug(f'Page: {page}  params: {params}')
            temp = getmeta(service='find', params=params, retry_http_error=True, logger=logger)
            # If there are no obs in the field (which is rare), None is returned
            if temp is not None:
                for row in temp:
                    obsids.append(row[0])
            else:
                temp = []
            page += 1
    return obsids
