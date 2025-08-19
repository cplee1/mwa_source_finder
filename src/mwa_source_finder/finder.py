import logging
import sys
from typing import Tuple

import yaml
from tqdm import tqdm

from .beam import source_beam_coverage
from .coordinates import get_atnf_pulsars, get_pointings
from .observations import (
    check_obsid_cache,
    get_all_obsids,
    get_common_metadata,
    save_as_yaml,
)

__all__ = ["find_sources_in_obs"]

logger = logging.getLogger(__name__)


def find_sources_in_obs(
    sources: list,
    obsids: list,
    t_start: float = 0.0,
    t_end: float = 1.0,
    obs_for_source: bool = False,
    condition: str = None,
    filter_available: bool = False,
    input_dt: float = 60.0,
    norm_mode: str = "zenith",
    min_power: float = 0.3,
    freq_mode: str = "centre",
    freq_samples: int = 10,
    no_cache: bool = False,
) -> Tuple[dict, dict]:
    """Find sources above a given power level in the MWA tile beam for a given
    observation.

    Parameters
    ----------
    sources : `list`
        A list of sources.
    obsids : `list`
        A list of obs IDs.
    t_start : `float`
        The start time to search, as a fraction of the full observation.
    t_end : `float`
        The end time to search, as a fraction of the full observation.
    obs_for_source : `bool`, optional
        Whether to search for observations for each source, by default False.
    condition : `str`, optional
        A condition to pass to the pulsar catalogue, by default None.
    filter_available : `bool`, optional
        Only search observations with data files available, by default False.
    input_dt : `float`, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : `str`, optional
        The normalisation mode ['zenith', 'beam'], by default 'zenith'.
    min_power : `float`, optional
        The minimum normalised power to count as in the beam, by default 0.3.
    freq_mode : `str`, optional
        The frequency to use to compute the beam power ['low', 'centre',
        'high'], by default 'centre'.
    freq_samples : `int`, optional
        If in multifreq mode, compute this many samples over the observing band,
        by default 10.
    no_cache : `bool`, optional
        Do not read or write to the metadata cache, by default False.

    Returns
    -------
    finder_results : `dict`
        A dictionary where each item [obsid/source] is a list of lists.
        The lists contain the obs ID/source name, the enter time, the
        exit time, and the maximum zenith-normalised power reached by
        the source in the beam.

    beam_coverage: `dict`
        A dictionary of dictionaries organised by obs IDs then source
        names, with each source entry is a list containing the enter
        time, the exit time, and the maximum zenith-normalised power
        reached by the source in the beam, and an array of powers for
        each time step.

    all_obs_metadata : `dict`
        A dictionary of dictionaries containing common metadata for each
        observation, organised by obs ID.

    pointings : `dict`
        A dictionary of dictionaries containing pointing information,
        organised by source name.
    """
    if sources is not None:
        # Convert source list to pointing list
        logger.info("Parsing pointings provided by user...")
        pointings = get_pointings(sources, condition=condition)
        # Print out a full list of sources
        logger.info(f"{len(pointings)} pointings parsed sucessfully")
        for source_name in pointings:
            pointing = pointings[source_name]
            logger.info(
                f"Source: {pointing['name']:30} "
                + f"RAJ: {pointing['RAJ']:14} "
                + f"DECJ: {pointing['DECJ']:15}"
            )
    else:
        logger.info("Collecting pulsars from the ATNF catalogue...")
        pointings = get_atnf_pulsars(condition=condition)
        logger.info(f"{len(pointings)} pulsar pointings parsed from the catalogue")

    if obsids is not None:
        # Get obs IDs from command line
        valid_obsids = []
        for obsid in obsids:
            if len(obsid) != 10:
                logger.error(f"Invalid obs ID provided: {obsid}")
                continue
            valid_obsids.append(obsid)
        logger.info(f"{len(obsids)} obs IDs parsed from user")
        logger.debug(f"Obs IDs: {obsids}")
    else:
        # Get all obs IDs
        logger.info("Obtaining a list of all obs IDs...")
        obsids = get_all_obsids()
        logger.info(f"{len(obsids)} obs IDs found in MWA archive")

    cached_obs_metadata = dict()
    all_obs_metadata = dict()
    obsids_to_query = obsids
    if not no_cache:
        # Check if the obs ID metadata has been cached
        cache_file = check_obsid_cache()
        if cache_file is not None:
            with open(cache_file, "r") as yamlfile:
                cached_obs_metadata = yaml.safe_load(yamlfile)
            obsids_to_query = []
            for req_obsid in obsids:
                req_obsid_found = False
                for cached_obsid in cached_obs_metadata.keys():
                    if cached_obsid == req_obsid:
                        req_obsid_found = True
                        all_obs_metadata[cached_obsid] = cached_obs_metadata[
                            cached_obsid
                        ]
                        break
                if not req_obsid_found:
                    obsids_to_query.append(req_obsid)
            logger.info(
                f"{len(obsids) - len(obsids_to_query)} obs IDs loaded from "
                + "cache file: {cache_file}"
            )

    # Only query the obs IDs which aren't in the cache
    if len(obsids_to_query) > 0:
        logger.info("Obtaining metadata for observations...")
        disable_tqdm = True
        if logger.level in [logging.DEBUG, logging.INFO]:
            disable_tqdm = False
        for obsid in tqdm(obsids_to_query, unit="obsid", disable=disable_tqdm):
            obs_metadata_tmp = get_common_metadata(obsid, filter_available)
            if obs_metadata_tmp is not None:
                all_obs_metadata[obsid] = obs_metadata_tmp
                cached_obs_metadata[obsid] = obs_metadata_tmp
        if not no_cache:
            # Update the cache file
            save_as_yaml(cached_obs_metadata)

    obsids = all_obs_metadata.keys()

    if len(obsids) == 0:
        logger.info("No observations available.")
        sys.exit(1)

    logger.info("Finding sources in beams...")
    beam_coverage, all_obs_metadata = source_beam_coverage(
        pointings,
        obsids,
        all_obs_metadata,
        t_start=t_start,
        t_end=t_end,
        input_dt=input_dt,
        norm_mode=norm_mode,
        min_power=min_power,
        freq_mode=freq_mode,
        freq_samples=freq_samples,
    )
    obsids = beam_coverage.keys()

    if not beam_coverage:
        logger.info("No sources found in beams.")
        sys.exit(1)

    finder_results = dict()
    if obs_for_source:
        for source_name in pointings:
            pointing = pointings[source_name]
            source_data = []
            for obsid in obsids:
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _, _ = beam_coverage[obsid][
                        source_name
                    ]
                    source_data.append([obsid, enter_beam, exit_beam, max_power])
            finder_results[source_name] = source_data
    else:
        for obsid in obsids:
            obsid_data = []
            for source_name in pointings:
                dm, p0 = None, None
                if "DM" in pointings[source_name]:
                    dm = pointings[source_name]["DM"]
                if "P0" in pointings[source_name]:
                    p0 = pointings[source_name]["P0"]
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _, _ = beam_coverage[obsid][
                        source_name
                    ]
                    obsid_data.append(
                        [source_name, enter_beam, exit_beam, max_power, dm, p0]
                    )
            finder_results[obsid] = obsid_data

    return finder_results, beam_coverage, pointings, all_obs_metadata
