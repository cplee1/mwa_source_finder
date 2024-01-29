import logging
import os
import sys
from typing import Tuple

import numpy as np
from scipy import interpolate
import mwa_hyperbeam

from mwa_source_finder import logger_setup, obs_utils, coord_utils


def get_beam_power_over_time(
    pointings: list,
    obsid: int,
    metadata: dict,
    offset: int = 0,
    dt: float = 60.0,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Compute the beam power towards a list of sources for a specified
    observation with a specified time resolution.

    Parameters
    ----------
    pointings : list
        A list of pointing dictionaries containing at minimum:
            RAJD : float
                The J2000 right ascension in decimal degrees.
            DECJD : float
                The J2000 declination in decimal degrees.
    obsid : int
        The observation ID.
    metadata : dict
        A dictionary of metadata containing at minimum:

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

    offset : int, optional
        Offset from the start of the observation in seconds, by default 0.
    dt : float, optional
        The step size in time to evaluate the power at, by default 60.
    norm_to_zenith : bool, optional
        Whether to normalise to the power at zenith, by default True.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    np.ndarray
        A 3D array of powers for each pointing, timestep, and channel.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    if os.environ.get("MWA_BEAM_FILE"):
        beam = mwa_hyperbeam.FEEBeam()
    else:
        logger.error(
            "MWA_BEAM_FILE environment variable not set! Please set to the "
            + "location of 'mwa_full_embedded_element_pattern.h5' file"
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


def beam_enter_exit(
    powers: np.ndarray,
    duration: int,
    dt: float = 60.0,
    min_z_power: float = 0.3,
    logger: logging.Logger = None,
) -> Tuple[float, float]:
    """Find where a source enters and exits the beam.

    Parameters
    ----------
    powers : np.ndarray
        A 2D array of powers for each timestep and channel.
    duration : int
        The observation duration in seconds.
    dt : float, optional
        The step size in time, by default 60.
    min_z_power : float, optional
        The minimum zenith-normalised power, by default 0.3.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the following:

            enter_beam : float
                The fraction of the observation where the source enters the beam.
            exit_beam : float
                The fraction of the observation where the source exits the beam.
    """
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
    pointings: list,
    obsids: list,
    obs_metadata_dict: dict,
    offset: int = 0,
    input_dt: float = 60.0,
    min_z_power: float = 0.3,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> dict:
    """For lists of pointings and observations, find where each source each
    source enters and exits each beam.

    Parameters
    ----------
    pointings : list
        A list of pointing dictionaries containing at minimum:

            Name : str
                The source name.
            RAJD : float
                The J2000 right ascension in decimal degrees.
            DECJD : float
                The J2000 declination in decimal degrees.

    obsids : list
        A list of observation IDs.
    obs_metadata_dict : dict
        A dictionary of metadata dictionaries (obs ID keys), each with at minimum:

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

    offset : int, optional
        Offset from the start of the observation in seconds, by default 0.
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    min_z_power : float, optional
        The minimum zenith-normalised power, by default 0.3.
    norm_to_zenith : bool, optional
        Whether to normalise to the power at zenith, by default True.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    dict
        A dictionary organised by obs IDs then source names, with each source
        having a list the following:
            enter_beam : float
                The fraction of the observation where the source enters the beam.
            exit_beam : float
                The fraction of the observation where the source exits the beam.
            max_pow: float
                The maximum power reached within the beam.
    """
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
    sources: list,
    obsids: list,
    obs_for_source: bool = False,
    offset: int = 0,
    input_dt: float = 60.0,
    min_z_power: float = 0.3,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> Tuple[dict, dict]:
    """Find sources in observations.

    Parameters
    ----------
    sources : list
        A list of sources.
    obsids : list
        A list of obs IDs.
    obs_for_source : bool, optional
        Whether to search for observations for each source, by default False.
    offset : int, optional
        Offset from the start of the observation in seconds, by default 0.
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    min_z_power : float, optional
        The minimum zenith-normalised power, by default 0.3.
    norm_to_zenith : bool, optional
        Whether to normalise to the power at zenith, by default True.
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

    logger.info("Obtaining metadata for observations...")
    obs_metadata_dict = dict()
    for obsid in obsids:
        obs_metadata_dict[obsid] = obs_utils.get_common_metadata(obsid, logger)

    logger.info("Finding sources in beams...")
    beam_coverage = source_beam_coverage(
        pointings,
        obsids,
        obs_metadata_dict,
        offset=offset,
        input_dt=input_dt,
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
                    source_data.append([obsid, enter_beam, exit_beam, max_power])
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
