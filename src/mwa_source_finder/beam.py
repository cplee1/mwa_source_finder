import logging
import os
import sys
from typing import Tuple

import mwa_hyperbeam
import numpy as np
from scipy import interpolate

import mwa_source_finder as sf

__all__ = [
    "compute_beam_power_array",
    "get_beam_power_vs_time",
    "get_beam_power_sky_map",
    "beam_enter_exit",
    "source_beam_coverage",
]


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
    az : `np.ndarray`
        The azimuth angles in radians.
    za : `np.ndarray`
        The zenith angles in radians.
    freq : `float`
        The frequency of the beam model.
    delays : `np.ndarray`
        The antenna delays.
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.

    Returns
    -------
    P : `np.ndarray`
        The beam powers as flattened array.
    """
    if logger is None:
        logger = sf.utils.get_logger()

    if os.environ.get("MWA_BEAM_FILE"):
        beam = mwa_hyperbeam.FEEBeam(None)
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
    freq_samples: int = 10,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the beam power over time for a multiple sources.

    Parameters
    ----------
    pointings : `dict`
        A dictionary of dictionaries containing pointing information, organised
        by source name.
    obs_metadata : `dict`
        A dictionary of commonly used metadata.
    t_start : `float`
        The start time to search, as a fraction of the full observation.
    t_end : `float`
        The end time to search, as a fraction of the full observation.
    input_dt : `float`
        The input step size in time (may be reduced).
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    freq_mode : `str`, optional
        The frequency to use to compute the beam power ['low', 'centre', 'high', 'multi'],
        by default 'centre'.
    freq_samples : `int`, optional
        If in multifreq mode, compute this many samples over the observing band,
        by default 10.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.

    Returns
    -------
    powers : `np.ndarray`
        An array of powers with dimensions of (#sources, #timesteps, #freqs).
    times : `np.ndarray`
        The timesteps used to compute the powers.
    duration : `float`
        The length of time searched in the observation.
    freq : `float`
        The frequency of the beam model used.
    """
    if logger is None:
        logger = sf.utils.get_logger()

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
        freqs = [1.28e6 * np.min(obs_metadata["channels"])]
    elif freq_mode == "centre":
        freqs = [1e6 * obs_metadata["centrefreq"]]
    elif freq_mode == "high":
        freqs = [1.28e6 * np.max(obs_metadata["channels"])]
    elif freq_mode == "multi":
        freqs = np.linspace(
            1.28e6 * np.min(obs_metadata["channels"]),
            1.28e6 * np.max(obs_metadata["channels"]),
            num=freq_samples,
        ).tolist()

    # Load the RA/DEC into numpy arrays
    RAs = np.empty(shape=(len(pointings)), dtype=float)
    DECs = np.empty(shape=(len(pointings)), dtype=float)
    for isource, source_name in enumerate(pointings):
        pointing = pointings[source_name]
        RAs[isource] = pointing["RAJD"]
        DECs[isource] = pointing["DECJD"]

    # Compute power for each source at each timestep
    num_coords = len(pointings) * len(times)
    Azs = np.zeros(num_coords, dtype=float)
    ZAs = np.zeros(num_coords, dtype=float)
    idx_step = len(times)
    for itime, time in enumerate(times):
        idx_start = itime
        idx_end = itime + num_coords
        _, Azs[idx_start:idx_end:idx_step], ZAs[idx_start:idx_end:idx_step] = sf.utils.equatorial_to_horizontal(
            RAs, DECs, time
        )

    powers_temp = np.zeros(num_coords, dtype=float)
    powers = np.empty(shape=(len(pointings), len(times), len(freqs)), dtype=float)
    for ifreq, freq in enumerate(freqs):
        logger.debug(f"Computing beam powers over time at freq = {freq/1e6:.2f} MHz")
        powers_temp = compute_beam_power_array(
            np.radians(Azs),
            np.radians(ZAs),
            freq,
            obs_metadata["delays"],
            norm_to_zenith=norm_to_zenith,
            logger=logger,
        )
        powers[:, :, ifreq] = powers_temp.reshape((len(pointings), len(times)))

    return powers, times, duration, freqs


def get_beam_power_sky_map(
    obs_metadata: dict,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the beam power for a grid of Az/ZA.

    Parameters
    ----------
    obs_metadata : `dict`
        A dictionary of commonly used metadata.
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.

    Returns
    -------
    az : `np.ndarray`
        The meshgrid of azimuth angles in radians.
    za : `np.ndarray`
        The meshgrid of zenith angles in radians.
    P : `np.ndarray`
        An array of powers in the shape of the meshgrid.
    """
    if logger is None:
        logger = sf.utils.get_logger()

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
        np.mean(obs_metadata["evalfreqs"]),
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
    powers : `np.ndarray`
        A 2D array of powers for each timestep and channel.
    times : `np.ndarray`
        The timesteps used to compute the powers.
    duration : `float`
        The length of time searched in the observation.
    min_power : `float`, optional
        The minimum power to count as in the beam. By default 0.3.

    Returns
    -------
    enter_beam : `float`
        The fraction of the observation where the source enters the beam.
    exit_beam : `float`
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
    freq_samples: int = 10,
    logger: logging.Logger = None,
) -> Tuple[dict, dict]:
    """For lists of pointings and observations, find where each source each
    source enters and exits each beam.

    Parameters
    ----------
    pointings : `dict`
        A dictionary of dictionaries containing pointing information,
        organised by source name.
    obsids : `list`
        A list of obs IDs.
    all_obs_metadata : `dict`
        A dictionary of dictionaries containing common metadata for each
        observation, organised by obs ID.
    t_start : `float`, optional
        The start time to search, as a fraction of the full observation, by
        default 0.0.
    t_end : `float`, optional
        The end time to search, as a fraction of the full observation, by
        default 1.0.
    input_dt : `float`, optional
        The input step size in time (may be reduced), by default 60.
    norm_mode : `str`, optional
        The normalisation mode, by default 'zenith'.
    min_power : `float`, optional
        The minimum normalised power to count as in the beam, by default 0.3.
    freq_mode : `str`, optional
        The frequency to use to compute the beam power ['low', 'centre', 'high', 'multi'],
        by default 'centre'.
    freq_samples : `int`, optional
        If in multifreq mode, compute this many samples over the observing band,
        by default 10.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.

    Returns
    -------
    beam_coverage : `dict`
        A dictionary organised by obs IDs then source names, with each source
        having a list the following:

            enter_beam : `float`
                The fraction of the observation where the source enters the beam.
            exit_beam : `float`
                The fraction of the observation where the source exits the beam.
            max_pow: `float`
                The maximum power reached within the beam.
            powers: `np.ndarray`
                An array of powers with dimensions of (#times, #freqs).
            times: `np.ndarray`
                The times at which the powers were evaluated.

    all_obs_metadata : `dict`
        The same as the input dictionary with an added 'evalfreqs' field for
        the frequencies at which each observation was searched.
    """
    if logger is None:
        logger = sf.utils.get_logger()

    if norm_mode == "zenith":
        norm_to_zenith = True
    else:
        norm_to_zenith = False

    if t_start > t_end:
        logger.critical("Selected start time is after selected end time.")
        sys.exit(1)

    beam_coverage = dict()
    obsids_to_remove = []
    for obsid in obsids:
        beam_coverage[obsid] = dict()
        obs_metadata = all_obs_metadata[obsid]

        logger.debug(f"Obs ID {obsid}: Getting beam powers")
        powers, times, duration, freqs = get_beam_power_vs_time(
            pointings,
            obs_metadata,
            t_start,
            t_end,
            input_dt,
            norm_to_zenith=norm_to_zenith,
            freq_mode=freq_mode,
            freq_samples=freq_samples,
            logger=logger,
        )
        all_obs_metadata[obsid]["evalfreqs"] = freqs

        logger.debug(f"Obs ID {obsid}: Getting enter and exit times")
        # Use only the lowest frequency in the powers array
        for isource, (source_power, source_name) in enumerate(zip(powers[:, :, 0], pointings)):
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
                    powers[isource, :, :],
                    times - float(obsid),
                ]
        if not beam_coverage[obsid]:
            obsids_to_remove.append(obsid)

    for obsid in obsids_to_remove:
        beam_coverage.pop(obsid)
        all_obs_metadata.pop(obsid)

    return beam_coverage, all_obs_metadata
