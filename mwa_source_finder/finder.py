import os
import sys

import numpy as np
from scipy import interpolate
import mwa_hyperbeam

from mwa_source_finder import logger_setup, obs_utils, coord_utils


def get_beam_power_over_time(
    pointings, obsid, metadata, offset=0, dt=60, norm_to_zenith=True, logger=None
):
    if logger is None:
        logger = logger_setup.get_logger()

    if os.environ.get("MWA_BEAM_FILE"):
        beam = mwa_hyperbeam.FEEBeam()
    else:
        logger.error(
            "MWA_BEAM_FILE environment variable not set! Please set to location of 'mwa_full_embedded_element_pattern.h5' file"
        )
        sys.exit(1)

    # Work out the time steps to model
    start_times = np.arange(offset, metadata["duration"] + offset, dt)
    stop_times = start_times + dt
    stop_times[stop_times > metadata["duration"]] = metadata["duration"]
    centre_times = float(obsid) + 0.5 * (start_times + stop_times)

    # Work out the frequencies to model
    frequencies = 1.28e6 * np.array(metadata["channels"], dtype=float)

    # Initialise the powers
    powers_x = np.zeros(
        shape=(len(pointings), len(start_times), len(frequencies)), dtype=float
    )
    powers_y = np.zeros(
        shape=(len(pointings), len(start_times), len(frequencies)), dtype=float
    )

    # Initialise the amplitudes
    amps = [1.0] * 16

    # Load the coordinates into numpy arrays
    ra_arr = np.empty(shape=(len(pointings)), dtype=float)
    dec_arr = np.empty(shape=(len(pointings)), dtype=float)
    for pi, pointing in enumerate(pointings):
        ra_arr[pi] = pointing["RAJD"]
        dec_arr[pi] = pointing["DECJD"]

    for itime in range(len(start_times)):
        _, az_arr, za_arr = coord_utils.equatorial_to_horizontal(
            ra_arr, dec_arr, centre_times[itime]
        )
        theta_arr = np.radians(za_arr)
        phi_arr = np.radians(az_arr)
        for ifreq in range(len(frequencies)):
            jones = beam.calc_jones_array(
                phi_arr,
                theta_arr,
                int(frequencies[ifreq]),
                metadata["delays"][0],
                amps,
                norm_to_zenith,
            )
            jones = jones.reshape(1, len(phi_arr), 2, 2)
            # Compute the visibility matrix using einstein summation
            vis = np.einsum("...ij,...kj->...ik", jones, np.conj(jones))
            rX, rY = (vis[:, :, 0, 0].real, vis[:, :, 1, 1].real)

            powers_x[:, itime, ifreq] = rX
            powers_y[:, itime, ifreq] = rY
    return 0.5 * (powers_x + powers_y)


def beam_enter_exit(powers, duration, dt=60, min_z_power=0.3, logger=None):
    if logger is None:
        logger = logger_setup.get_logger()

    time_steps = np.array(np.arange(0, duration, dt), dtype=float)
    powers_freq_min = np.empty(shape=(len(powers)), dtype=float)
    for pi, p in enumerate(powers):
        powers_freq_min[pi] = float(min(p) - min_z_power)
    if np.min(powers_freq_min) > 0.0:
        enter_beam = 0.0
        exit_beam = 1.0
    else:
        spline = interpolate.UnivariateSpline(time_steps, powers_freq_min, s=0.0)
        # try:
        #     spline = interpolate.UnivariateSpline(time_steps, powers_freq_min, s=0.0)
        # except ValueError:
        #     logger.error("Could not fit Univariate Spline")
        #     return None, None
        if len(spline.roots()) == 2:
            enter_beam, exit_beam = spline.roots()
            enter_beam /= duration
            exit_beam /= duration
        elif len(spline.roots()) == 1:
            if powers_freq_min[0] > powers_freq_min[-1]:
                enter_beam = 0.0
                exit_beam = spline.roots()[0] / duration
            else:
                enter_beam = spline.roots()[0] / duration
                exit_beam = 1.0
        else:
            enter_beam = 0.0
            exit_beam = 1.0
    return enter_beam, exit_beam


def source_beam_coverage(
    pointings,
    obsids,
    obs_metadata_dict,
    offset=0,
    input_dt=60,
    min_z_power=0.3,
    norm_to_zenith=True,
    logger=None,
):
    if logger is None:
        logger = logger_setup.get_logger()

    beam_coverage = dict()
    for obsid in obsids:
        beam_coverage[obsid] = dict()
        obs_metadata = obs_metadata_dict[obsid]

        if obs_metadata["duration"] / input_dt < 4:
            multiplier = 4 * input_dt / obs_metadata["duration"]
            dt = input_dt / multiplier
            logger.debug(f"Obs ID {obsid}: Using reduced dt={dt}")
        else:
            dt = input_dt

        logger.debug(f"Obs ID {obsid}: Getting beam powers")
        powers = get_beam_power_over_time(
            pointings,
            obsid,
            obs_metadata,
            offset=offset,
            dt=dt,
            norm_to_zenith=norm_to_zenith,
            logger=logger,
        )
        logger.debug(f"Obs ID {obsid}: Getting enter and exit times")
        for source_obs_power, pointing in zip(powers, pointings):
            if np.max(source_obs_power) > min_z_power:
                beam_enter, beam_exit = beam_enter_exit(
                    source_obs_power,
                    obs_metadata["duration"],
                    dt=dt,
                    min_z_power=min_z_power,
                    logger=logger,
                )
                beam_coverage[obsid][pointing["Name"]] = [
                    beam_enter,
                    beam_exit,
                    np.amax(source_obs_power),
                ]
    return beam_coverage


def find_sources_in_obs(
    sources,
    obsids,
    obs_for_source=False,
    offset=0,
    dt=60,
    min_z_power=0.3,
    norm_to_zenith=True,
    logger=None,
):
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

    logger.info('Obtaining metadata for observations...')
    obs_metadata_dict = dict()
    for obsid in obsids:
        obs_metadata_dict[obsid] = obs_utils.get_common_metadata(obsid, logger)

    logger.info('Finding sources in beams...')
    beam_coverage = source_beam_coverage(
        pointings,
        obsids,
        obs_metadata_dict,
        offset=offset,
        input_dt=dt,
        min_z_power=min_z_power,
        norm_to_zenith=norm_to_zenith,
        logger=logger,
    )

    output_data = dict()
    if obs_for_source:
        for pointing in pointings:
            source_name = pointing["Name"]
            source_data = []
            for obsid in obsids:
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power = beam_coverage[obsid][source_name]
                    obs_metadata = obs_metadata_dict[obsid]
                    source_data.append(
                        [
                            obsid,
                            enter_beam,
                            exit_beam,
                            max_power,
                            obs_metadata["duration"],
                            obs_metadata["centrefreq"],
                            obs_metadata["bandwidth"],
                        ]
                    )
            output_data[source_name] = source_data
    else:
        for obsid in obsids:
            obsid_data = []
            for pointing in pointings:
                source_name = pointing["Name"]
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power = beam_coverage[obsid][source_name]
                    obsid_data.append([source_name, enter_beam, exit_beam, max_power])
            output_data[obsid] = obsid_data
    return output_data, obs_metadata_dict
