import logging
import os
import sys
from typing import Tuple

import numpy as np
from scipy import interpolate
import mwa_hyperbeam

from mwa_source_finder import logger_setup, coord_utils


def compute_beam_power_array(
    az: np.ndarray,
    za: np.ndarray,
    freq: float,
    delays: np.ndarray,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> np.ndarray:
    """Compute the beam power for an array of sky coordinates.

    Parameters
    ----------
    az : np.ndarray
        The azimuth angles in radians.
    za : np.ndarray
        The zenith angles in radians.
    freq : float
        The frequency of the beam model.
    delays : np.ndarray
        The antenna delays.
    norm_to_zenith : bool, optional
        Whether to normalise the powers to zenith, by default True.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    np.ndarray
        The beam powers as flattened array.
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

    jones = beam.calc_jones_array(
        az.flatten(),
        za.flatten(),
        freq,
        delays,
        np.ones_like(delays),
        norm_to_zenith,
    )

    # Define the sky matrix
    S = np.eye(2) / 2

    # Define the Jones matrix and its conjugate transpose
    J = jones.reshape(-1, 2, 2)
    K = np.conjugate(J).T

    # Compute the sky power
    P = np.einsum("Nki,ij,jkN->N", J, S, K, optimize=True).real

    return P


def get_beam_power_vs_time(
    pointings: dict,
    obs_metadata: dict,
    t_start: float,
    t_end: float,
    input_dt: float,
    norm_to_zenith: bool = True,
    freq_mode: str = "centre",
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the beam power over time for a multiple sources.

    Parameters
    ----------
    pointings : dict
        A dictionary of dictionaries containing pointing information, organised
        by source name.
    obs_metadata : dict
        A dictionary of commonly used metadata.
    t_start : float
        The start time to search, as a fraction of the full observation.
    t_end : float
        The end time to search, as a fraction of the full observation.
    input_dt : float
        The input step size in time (may be reduced).
    norm_to_zenith : bool, optional
        Whether to normalise the powers to zenith, by default True.
    freq_mode : str, optional
        The frequency to use to compute the beam power ['low', 'centre', 'high'],
        by default 'centre'.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float]
        A tuple containing the following:

            P : np.ndarray
                An array of powers with dimensions of (#sources, #timesteps).
            times : np.ndarray
                The timesteps used to compute the powers.
            duration : float
                The length of time searched in the observation.
            freq : float
                The frequency of the beam model used.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    # Unpack some metadata
    obsid = obs_metadata["obsid"]
    full_duration = obs_metadata["duration"]

    # Lower dt if necessary
    duration = (t_end - t_start) * full_duration
    if duration / input_dt < 4:
        multiplier = 4 * input_dt / duration
        dt = input_dt / multiplier
        logger.debug(f"Obs ID {obsid}: Using reduced dt={dt}")
    else:
        dt = input_dt

    # Choose time steps to model
    t_start_sec = float(obsid) + t_start * full_duration
    t_end_sec = float(obsid) + t_end * full_duration
    start_times = np.arange(t_start_sec, t_end_sec, dt)
    times = np.append(start_times, t_end_sec)

    # Choose frequency to model
    if freq_mode == "low":
        freq = 1.28e6 * np.min(obs_metadata["channels"])
    elif freq_mode == "centre":
        freq = 1e6 * obs_metadata["centrefreq"]
    elif freq_mode == "high":
        freq = 1.28e6 * np.max(obs_metadata["channels"])

    # Load the RA/DEC into numpy arrays
    RAs = np.empty(shape=(len(pointings)), dtype=float)
    DECs = np.empty(shape=(len(pointings)), dtype=float)
    for isource, source_name in enumerate(pointings):
        pointing = pointings[source_name]
        RAs[isource] = pointing["RAJD"]
        DECs[isource] = pointing["DECJD"]

    # Compute power for each source at each timestep
    P = np.zeros(shape=(len(pointings), len(times)), dtype=float)
    for itime, time in enumerate(times):
        _, Azs, ZAs = coord_utils.equatorial_to_horizontal(RAs, DECs, time)
        logger.debug(f"Time step {itime}: Computing beam power")
        P[:, itime] = compute_beam_power_array(
            np.radians(Azs),
            np.radians(ZAs),
            freq,
            obs_metadata["delays"],
            norm_to_zenith=norm_to_zenith,
            logger=logger,
        )
    return P, times, duration, freq


def get_beam_power_sky_map(
    obs_metadata: dict,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the beam power for a grid of Az/ZA.

    Parameters
    ----------
    obs_metadata : dict
        A dictionary of commonly used metadata.
    norm_to_zenith : bool, optional
        Whether to normalise the powers to zenith, by default True.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the following:

            az : np.ndarray
                The meshgrid of azimuth angles in radians.
            za : np.ndarray
                The meshgrid of zenith angles in radians.
            P : np.ndarray
                An array of powers in the shape of the meshgrid.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    az0, az1 = 0, 2 * np.pi
    za0, za1 = 0, 0.95 * np.pi / 2
    grid_res = 600
    _az = np.linspace(az0, az1, grid_res)
    _za = np.linspace(za0, za1, grid_res)
    az, za = np.meshgrid(_az, _za)

    logger.debug(f"Obs ID {obs_metadata['obsid']}: Computing beam power")
    P = compute_beam_power_array(
        az,
        za,
        obs_metadata["evalfreq"],
        obs_metadata["delays"],
        norm_to_zenith=norm_to_zenith,
        logger=logger,
    )
    return az, za, P.reshape(az.shape)


def beam_enter_exit(
    powers: np.ndarray,
    times: np.ndarray,
    duration: float,
    min_power: float = 0.3,
) -> Tuple[float, float]:
    """Find where a source enters and exits the beam.

    Parameters
    ----------
    powers : np.ndarray
        A 2D array of powers for each timestep and channel.
    times : np.ndarray
        The timesteps used to compute the powers.
    duration : float
        The length of time searched in the observation.
    min_power : float, optional
        The minimum power to count as in the beam. By default 0.3.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the following:

            enter_beam : float
                The fraction of the observation where the source enters the beam.
            exit_beam : float
                The fraction of the observation where the source exits the beam.
    """
    powers_offset = powers - min_power

    if np.min(powers_offset) > 0.0:
        enter_beam = 0.0
        exit_beam = 1.0
    else:
        spline = interpolate.UnivariateSpline(times, powers_offset, s=0.0)
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
    pointings: dict,
    obsids: list,
    all_obs_metadata: dict,
    t_start: float = 0.0,
    t_end: float = 1.0,
    input_dt: float = 60.0,
    norm_mode: str = "zenith",
    min_power: float = 0.3,
    freq_mode: str = "centre",
    logger: logging.Logger = None,
) -> Tuple[dict, dict]:
    """For lists of pointings and observations, find where each source each
    source enters and exits each beam.

    Parameters
    ----------
    pointings : dict
        A dictionary of dictionaries containing pointing information,
        organised by source name.
    obsids : list
        A list of obs IDs.
    all_obs_metadata : dict
        A dictionary of dictionaries containing common metadata for each
        observation, organised by obs ID.
    t_start : float, optional
        The start time to search, as a fraction of the full observation, by
        default 0.0.
    t_end : float, optional
        The end time to search, as a fraction of the full observation, by
        default 1.0.
    input_dt : float, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : str, optional
        The normalisation mode, by default 'zenith'.
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

        beam_coverage : dict
            A dictionary organised by obs IDs then source names, with each source
            having a list the following:

                enter_beam : float
                    The fraction of the observation where the source enters the beam.
                exit_beam : float
                    The fraction of the observation where the source exits the beam.
                max_pow: float
                    The maximum power reached within the beam.

        all_obs_metadata : dict
            The same as the input dictionary with an added 'evalfreq' field for
            the frequency at which each observation was searched.
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
        obs_metadata = all_obs_metadata[obsid]

        logger.debug(f"Obs ID {obsid}: Getting beam powers")
        powers, times, duration, freq = get_beam_power_vs_time(
            pointings,
            obs_metadata,
            t_start,
            t_end,
            input_dt,
            norm_to_zenith=norm_to_zenith,
            freq_mode=freq_mode,
            logger=logger,
        )
        all_obs_metadata[obsid]["evalfreq"] = freq

        logger.debug(f"Obs ID {obsid}: Getting enter and exit times")
        for source_power, source_name in zip(powers, pointings):
            if np.max(source_power) > min_power:
                beam_enter, beam_exit = beam_enter_exit(
                    source_power,
                    times - float(obsid),
                    duration,
                    min_power=min_power,
                )
                beam_coverage[obsid][source_name] = [
                    beam_enter,
                    beam_exit,
                    np.amax(source_power),
                    source_power,
                    times - float(obsid),
                ]
        if not beam_coverage[obsid]:
            beam_coverage.pop(obsid)
            all_obs_metadata.pop(obsid)
    return beam_coverage, all_obs_metadata
