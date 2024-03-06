import logging
import sys
from typing import Tuple

from mwa_source_finder import logger_setup, obs_utils, coord_utils, beam_1D


def find_sources_in_obs(
    sources: list,
    obsids: list,
    t_start: float = 0.0,
    t_end: float = 1.0,
    obs_for_source: bool = False,
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
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : str, optional
        The normalisation mode ['zenith', 'beam'], by default 'zenith'.
    min_power : float, optional
        The minimum power to count as in the beam. If a normalisation mode is
        selected, then this will be interpreted as a normalised power. By default 0.3.
    freq_mode : str, optional
        The frequency to use to compute the beam power ['low', 'centre', 'high'],
        by default 'centre'.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing the following:

            output_data : dict
                A dictionary containing the results. If obs_for_source is False,
                the dictionary is ordered by obs ID and contains a list of
                lists, each with:

                    source_name : str
                        The source name.
                    enter_beam : float
                        The fraction of the observation where the source enters
                        the beam.
                    exit_beam : float
                        The fraction of the observation where the source exits
                        the beam.
                    max_power : float
                        The maximum power reached within the beam.

                If obs_for_source is True, the dictionary is ordered by source
                name and contains a list of lists, each with:

                    obsid : int
                        The observation ID.
                    enter_beam : float
                        The fraction of the observation where the source enters
                        the beam.
                    exit_beam : float
                        The fraction of the observation where the source exits
                        the beam.
                    max_power: float
                        The maximum power reached within the beam.

            obs_metadata_dict : dict
                A dictionary containing common metadata for each observation,
                organised by observation ID, each with the following items:

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

            pointings : list
                A dictionary of dictionaries, organised by source name, each
                with the following items:

                    RAJ : str
                        The J2000 right ascension in sexigesimal format.
                    DEC : str
                        The J2000 declination in sexigesimal format.
                    RAJD : float
                        The J2000 right ascension in decimal degrees.
                    DECJD : float
                        The J2000 declination in decimal degrees.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    if sources is not None:
        # Convert source list to pointing list
        pointings = coord_utils.get_pointings(sources, logger=logger)
        # Print out a full list of sources
        logger.info(f"{len(pointings)} pointings parsed sucessfully")
        for pointing in pointings:
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
            if len(str(obsid)) != 10:
                logger.error(f"Invalid obs ID provided: {obsid}")
                continue
            valid_obsids.append(obsid)
        # Print out a full list of observations
        logger.info(f"Obs IDs: {obsids}")
    else:
        logger.info("Obtaining a list of all obs IDs...")
        obsids = obs_utils.get_all_obsids(logger=logger)
        logger.info(f"{len(obsids)} observations found")

    obs_metadata_dict = dict()
    for obsid in obsids:
        logger.debug(f"Obtaining metadata for obs ID: {obsid}")
        obs_metadata_tmp = obs_utils.get_common_metadata(obsid, logger)
        if obs_metadata_tmp is not None:
            obs_metadata_dict[obsid] = obs_metadata_tmp
    obsids = list(obs_metadata_dict)

    logger.info("Finding sources in beams...")
    beam_coverage, obs_metadata_dict = beam_1D.source_beam_coverage(
        pointings,
        obsids,
        obs_metadata_dict,
        t_start=t_start,
        t_end=t_end,
        input_dt=input_dt,
        norm_mode=norm_mode,
        min_power=min_power,
        freq_mode=freq_mode,
        logger=logger,
    )
    obsids = list(beam_coverage)

    if not beam_coverage:
        logger.info("No sources found in beams.")
        sys.exit(1)

    output_data = dict()
    if obs_for_source:
        for pointing in pointings:
            source_name = pointing["name"]
            source_data = []
            for obsid in obsids:
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _ = beam_coverage[obsid][
                        source_name
                    ]
                    source_data.append([obsid, enter_beam, exit_beam, max_power])
            output_data[source_name] = source_data
    else:
        for obsid in obsids:
            obsid_data = []
            for pointing in pointings:
                source_name = pointing["name"]
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _ = beam_coverage[obsid][
                        source_name
                    ]
                    obsid_data.append([source_name, enter_beam, exit_beam, max_power])
            output_data[obsid] = obsid_data

    return output_data, beam_coverage, pointings, obs_metadata_dict
