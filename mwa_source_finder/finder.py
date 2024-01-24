import os
import sys

import numpy as np
import mwa_hyperbeam

import mwa_source_finder.logger_setup as logger_setup
import mwa_source_finder.obs_utils as obs_utils
import mwa_source_finder.coord_utils as coord_utils


def get_beam_power_over_time(pointings, obsid, metadata, offset=0, dt=100, logger=None):
    if logger is None:
        logger = logger_setup.get_logger()

    if os.environ.get("MWA_BEAM_FILE"):
        beam = mwa_hyperbeam.FEEBeam()
    else:
        logger.error(
            "MWA_BEAM_FILE environment variable not set! Please set to location of 'mwa_full_embedded_element_pattern.h5' file"
        )
        sys.exit(1)

    start_times = np.arange(offset, metadata["duration"] + offset, dt)
    stop_times = start_times + dt
    # No stop time can exceed the observation length
    stop_times[stop_times > metadata["duration"]] = metadata["duration"]
    centre_times = float(obsid) + 0.5 * (start_times + stop_times)

    frequencies = 1.28e6 * metadata["channels"]


def source_beam_coverage(pointings, obsids, logger=None):
    if logger is None:
        logger = logger_setup.get_logger()

    for obsid in obsids:
        obs_metadata = obs_utils.get_common_metadata(obsid, logger)


def find_sources_in_obs(sources, obsids, logger=None):
    if logger is None:
        logger = logger_setup.get_logger()

    if sources is not None:
        # Convert source list to pointing list
        pointings = coord_utils.get_pointings(sources, logger=logger)
        # Print out a full list of sources
        logger.info(f"{len(pointings)} pointings parsed sucessfully")
        for pointing in pointings:
            logger.info(
                f"Source: {pointing['Name']:30} "
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
        logger.info("Retrieving metadata for all observations...")
        obsids = obs_utils.get_all_obsids(logger=logger)
        logger.info(f"{len(obsids)} observations found")
        logger.debug(f"Obs IDs: {obsids}")
