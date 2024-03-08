import logging
from typing import Tuple

import numpy as np

from mwa_source_finder import logger_setup


def plan_obs_times(
    obs_metadata: dict,
    power: np.ndarray,
    times: np.ndarray,
    obs_length: float,
    logger: logging.Logger = None,
) -> Tuple[float, float]:
    """For arrays of power vs time, find obs_length seconds where the power is
    the highest, and return the start and stop times.

    Parameters
    ----------
    obs_metadata : dict
        A dictionary of commonly used metadata.
    power : np.ndarray
        The source powers.
    times : np.ndarray
        The times of the powers from the start of the observation, in seconds.
    obs_length : float
        The desired observation length, in seconds.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the following:

            start_t : float
                The start time of the optimal observation, in seconds.
            stop_t : float
                The stop time of the optimal observation, in seconds.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    # Unpack some metadata
    obsid = obs_metadata["obsid"]
    obs_duration = obs_metadata["duration"]

    # Check that the minimum obs length is smaller than the obs duration
    if obs_length > obs_duration:
        logger.error(
            f"Length of obs {obsid} is shorter than the minimum obs length of "
            + f"{obs_length:.2f} s. Assuming full obs length."
        )
        return 0.0, float(obs_duration)
    elif obs_length == obs_duration:
        return 0.0, float(obs_duration)

    # If the peak is at the edge, return the min obs length from the edge.
    # Otherwise, fit a parabola to the peak and find the nearest obs_length
    # assuming that the power level is symmetrical about the peak.
    peak_idx = np.argmax(power)
    if peak_idx == 0:
        start_t = 0.0
        stop_t = obs_length
    elif peak_idx >= len(power) - 2:
        start_t = obs_duration - obs_length
        stop_t = obs_duration
    else:
        y = np.abs(power[peak_idx - 1 : peak_idx + 2])
        x = times[peak_idx - 1 : peak_idx + 2]
        p = np.polyfit(x, y, 2)
        peak_time = -p[1] / (2.0 * p[0])
        if peak_time < obs_length / 2:
            start_t = 0.0
            stop_t = obs_length
        elif obs_duration - peak_time < obs_length / 2:
            start_t = obs_duration - obs_length
            stop_t = obs_duration
        else:
            start_t = peak_time - obs_length / 2
            stop_t = peak_time + obs_length / 2
    return start_t, stop_t


# TODO: If the best observation is too short, use the second best, etc
def find_best_obs_times_for_sources(
    source_names: list,
    all_obs_metadata: dict,
    beam_coverage: dict,
    obs_length: float = None,
    logger: logging.Logger = None,
) -> dict:
    """Find the best observation for each source, based on the mean power level,
    then return the optimal start and stop times of an observation of obs_length
    seconds for each source.

    Parameters
    ----------
    source_names : list
        A list of source names.
    all_obs_metadata : dict
        A dictionary of metadata dictionaries.
    beam_coverage : dict
        A dictionary of dictionaries organised by obs IDs then source names,
        with each source entry is a list containing the enter time, the exit
        time, and the maximum zenith-normalised power reached by the source in
        the beam, and an array of powers for each time step.
    obs_length : float
        The desired observation length, in seconds.
    logger : logging.Logger, optional
        A custom logger to use, by default None.

    Returns
    -------
    dict
        A dictionary organised by source name, with each entry being a list
        containing the obs ID, start time, and stop time, of the best
        observation.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    best_obs_results = dict()
    for source_name in source_names:
        obsid_list, mean_power_list = [], []
        for obsid in all_obs_metadata:
            if source_name in beam_coverage[obsid]:
                _, _, _, source_power, times = beam_coverage[obsid][source_name]
                obsid_list.append(obsid)
                mean_power_list.append(np.mean(source_power))
        best_obsid = obsid_list[np.argmax(np.array(mean_power_list))]
        _, _, _, best_obsid_power, best_obsid_times = beam_coverage[best_obsid][
            source_name
        ]
        best_obs_metadata = all_obs_metadata[best_obsid]
        start_t, stop_t = plan_obs_times(
            best_obs_metadata,
            best_obsid_power,
            best_obsid_times,
            obs_length=obs_length,
            logger=logger,
        )
        best_obs_results[source_name] = [best_obsid, start_t, stop_t]
    return best_obs_results
