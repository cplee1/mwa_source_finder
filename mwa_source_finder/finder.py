import logging
import sys
from typing import Tuple

from mwa_source_finder import logger_setup, obs_utils, coord_utils, beam_utils


def find_sources_in_obs(
    sources: list,
    obsids: list,
    t_start: float = 0.0,
    t_end: float = 1.0,
    obs_for_source: bool = False,
    filter_available: bool = False,
    input_dt: float = 60.0,
    norm_mode: str = "zenith",
    min_power: float = 0.3,
    freq_mode: str = "centre",
    logger: logging.Logger = None,
) -> Tuple[dict, dict]:
    """Find sources in observations.

    Parameters
    ----------
    sources : list
        A list of sources.
    obsids : list
        A list of obs IDs.
    t_start : float
        The start time to search, as a fraction of the full observation.
    t_end : float
        The end time to search, as a fraction of the full observation.
    obs_for_source : bool, optional
        Whether to search for observations for each source, by default False.
    filter_available : bool, optional
        Only search observations with data files available, by default False.
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : str, optional
        The normalisation mode ['zenith', 'beam'], by default 'zenith'.
    min_power : float, optional
        The minimum normalised power to count as in the beam, by default 0.3.
    freq_mode : str, optional
        The frequency to use to compute the beam power ['low', 'centre', 'high'],
        by default 'centre'.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing the following:

            finder_results : dict
                A dictionary where each item [obsid/source] is a list of lists.
                The lists contain the obs ID/source name, the enter time, the
                exit time, and the maximum zenith-normalised power reached by
                the source in the beam.

            beam_coverage: dict
                A dictionary of dictionaries organised by obs IDs then source
                names, with each source entry is a list containing the enter
                time, the exit time, and the maximum zenith-normalised power
                reached by the source in the beam, and an array of powers for
                each time step.

            all_obs_metadata : dict
                A dictionary of dictionaries containing common metadata for each
                observation, organised by obs ID.

            pointings : dict
                A dictionary of dictionaries containing pointing information,
                organised by source name.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    if sources is not None:
        # Convert source list to pointing list
        pointings = coord_utils.get_pointings(sources, logger=logger)
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
        pointings = coord_utils.get_atnf_pulsars(logger=logger)
        logger.info(f"{len(pointings)} pulsar pointings parsed from the catalogue")

    if obsids is not None:
        # Get obs IDs from command line
        valid_obsids = []
        for obsid in obsids:
            if len(obsid) != 10:
                logger.error(f"Invalid obs ID provided: {obsid}")
                continue
            valid_obsids.append(obsid)
        # Print out a full list of observations
        logger.info(f"Obs IDs: {obsids}")
    else:
        logger.info("Obtaining a list of all obs IDs...")
        obsids = obs_utils.get_all_obsids(logger=logger)
        logger.info(f"{len(obsids)} observations found")

    all_obs_metadata = dict()
    logger.info("Obtaining metadata for observations...")
    for obsid in obsids:
        logger.debug(f"Obtaining metadata for obs ID: {obsid}")
        obs_metadata_tmp = obs_utils.get_common_metadata(
            obsid, filter_available, logger
        )
        if obs_metadata_tmp is not None:
            all_obs_metadata[obsid] = obs_metadata_tmp
    obsids = all_obs_metadata.keys()

    if len(obsids) == 0:
        logger.info("No observations available.")
        sys.exit(1)

    logger.info("Finding sources in beams...")
    beam_coverage, all_obs_metadata = beam_utils.source_beam_coverage(
        pointings,
        obsids,
        all_obs_metadata,
        t_start=t_start,
        t_end=t_end,
        input_dt=input_dt,
        norm_mode=norm_mode,
        min_power=min_power,
        freq_mode=freq_mode,
        logger=logger,
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
                pointing = pointings[source_name]
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _, _ = beam_coverage[obsid][
                        source_name
                    ]
                    obsid_data.append([source_name, enter_beam, exit_beam, max_power])
            finder_results[obsid] = obsid_data

    return finder_results, beam_coverage, pointings, all_obs_metadata
