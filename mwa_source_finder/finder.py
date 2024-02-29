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
    obs_metadata: dict,
    freq: float,
    times: list,
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
    obs_metadata : dict
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

    freq : float
        The frequency to compute the beam power at in Hz.
    times: list
        A list of times to compute the beam power at.
    norm_to_zenith : bool, optional
        Whether to normalise powers to zenith, by default True
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

    powers = np.zeros(shape=(len(pointings), len(times)), dtype=float)
    ra_arr = np.empty(shape=(len(pointings)), dtype=float)
    dec_arr = np.empty(shape=(len(pointings)), dtype=float)
    S = np.eye(2) / 2

    # Load the coordinates into numpy arrays
    for pi, pointing in enumerate(pointings):
        ra_arr[pi] = pointing["RAJD"]
        dec_arr[pi] = pointing["DECJD"]

    for itime in range(len(times)):
        _, az_arr, za_arr = coord_utils.equatorial_to_horizontal(
            ra_arr, dec_arr, times[itime]
        )

        jones = beam.calc_jones_array(
            np.radians(az_arr),
            np.radians(za_arr),
            freq,
            obs_metadata["delays"],
            np.ones_like(obs_metadata["delays"]),
            norm_to_zenith,
        )

        J = jones.reshape(-1, 2, 2)
        K = np.conjugate(J).T
        powers[:, itime] = np.einsum("Nki,ij,jkN->N", J, S, K, optimize=True).real

    return powers


def beam_enter_exit(
    powers: np.ndarray,
    duration: int,
    time_steps: list,
    min_power: float = 0.3,
    logger: logging.Logger = None,
) -> Tuple[float, float]:
    """Find where a source enters and exits the beam.

    Parameters
    ----------
    powers : np.ndarray
        A 2D array of powers for each timestep and channel.
    duration : int
        The observation duration in seconds.
    time_steps : list
        The time steps to compute the beam power at.
    min_power : float, optional
        The minimum power to count as in the beam. By default 0.3.
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

    powers_offset = powers - min_power

    if np.min(powers_offset) > 0.0:
        enter_beam = 0.0
        exit_beam = 1.0
    else:
        spline = interpolate.UnivariateSpline(time_steps, powers_offset, s=0.0)
        if len(spline.roots()) == 2:
            enter_beam, exit_beam = spline.roots()
            enter_beam /= duration
            exit_beam /= duration
        elif len(spline.roots()) == 1:
            if powers_offset[0] > powers_offset[-1]:
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
    t_start: float = 0.0,
    t_end: float = 1.0,
    input_dt: float = 60.0,
    norm_mode: str = "zenith",
    min_power: float = 0.3,
    freq_mode: str = "centre",
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

    t_start : float
        The start time to search, as a fraction of the full observation.
    t_end : float
        The end time to search, as a fraction of the full observation.
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : str, optional
        The normalisation mode, by default 'zenith'.
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

    if norm_mode == "zenith":
        norm_to_zenith = True
    else:
        norm_to_zenith = False

    if t_start > t_end:
        logger.critical("Selected start time is after selected end time.")
        sys.exit(1)

    beam_coverage = dict()
    for obsid in obsids:
        beam_coverage[obsid] = dict()
        obs_metadata = obs_metadata_dict[obsid]
        duration = (t_end - t_start) * obs_metadata["duration"]

        if duration / input_dt < 4:
            multiplier = 4 * input_dt / duration
            dt = input_dt / multiplier
            logger.debug(f"Obs ID {obsid}: Using reduced dt={dt}")
        else:
            dt = input_dt

        # Choose frequency to model
        if freq_mode == "low":
            freq = 1.28e6 * np.min(obs_metadata["channels"])
        elif freq_mode == "centre":
            freq = 1e6 * obs_metadata["centrefreq"]
        elif freq_mode == "high":
            freq = 1.28e6 * np.max(obs_metadata["channels"])

        # Choose time steps to model
        t_start_sec = float(obsid) + t_start * obs_metadata["duration"]
        t_end_sec = float(obsid) + t_end * obs_metadata["duration"]
        start_times = np.arange(t_start_sec, t_end_sec, dt)
        times = np.append(start_times, t_end_sec)

        logger.debug(f"Obs ID {obsid}: Getting beam powers")
        powers = get_beam_power_over_time(
            pointings,
            obs_metadata,
            freq,
            times,
            norm_to_zenith=norm_to_zenith,
            logger=logger,
        )

        logger.debug(f"Obs ID {obsid}: Getting enter and exit times")
        for source_obs_power, pointing in zip(powers, pointings):
            if np.max(source_obs_power) > min_power:
                beam_enter, beam_exit = beam_enter_exit(
                    source_obs_power,
                    duration,
                    times - float(obsid),
                    min_power=min_power,
                    logger=logger,
                )
                beam_coverage[obsid][pointing["Name"]] = [
                    beam_enter,
                    beam_exit,
                    np.amax(source_obs_power),
                    source_obs_power,
                ]
        if not beam_coverage[obsid]:
            beam_coverage.pop(obsid)
            obs_metadata_dict.pop(obsid)
    return beam_coverage, obs_metadata_dict, freq


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
    beam_coverage, obs_metadata_dict, freq = source_beam_coverage(
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
            source_name = pointing["Name"]
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
                source_name = pointing["Name"]
                if source_name in beam_coverage[obsid]:
                    enter_beam, exit_beam, max_power, _ = beam_coverage[obsid][
                        source_name
                    ]
                    obsid_data.append([source_name, enter_beam, exit_beam, max_power])
            output_data[obsid] = obsid_data

    return output_data, beam_coverage, pointings, obs_metadata_dict, freq
