import csv
import logging
from typing import Tuple

import numpy as np

__all__ = [
    "plan_obs_times",
    "find_best_obs_times_for_sources",
    "plan_data_download",
    "find_contiguous_ranges",
]

logger = logging.getLogger(__name__)


def _round_down(time: float, chunksize: float = 8.0) -> float:
    """Round the time down to the nearest multiple of the chunksize.

    Parameters
    ----------
    time : `float`
        The time to round.
    chunksize : `float`, optional
        The multiple to round to, in the same units as the time, by default 8.

    Returns
    -------
    time : `float`
        The rounded time.
    """
    return time - (time % chunksize)


def plan_obs_times(obs_metadata: dict, power: np.ndarray, times: np.ndarray, obs_length: float) -> Tuple[float, float]:
    """For arrays of power vs time, find obs_length seconds where the power is
    the highest, and return the start and stop times.

    Parameters
    ----------
    obs_metadata : `dict`
        A dictionary of commonly used metadata.
    power : `np.ndarray`
        The source powers.
    times : `np.ndarray`
        The times of the powers from the start of the observation, in seconds.
    obs_length : `float`
        The desired observation length, in seconds.

    Returns
    -------
    start_t : `float`
        The start time of the optimal observation, in seconds.
    stop_t : `float`
        The stop time of the optimal observation, in seconds.
    peak_time : `float`
        The time of the peak power in the beam, in seconds.
    """
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
        peak_time = 0.0
    elif peak_idx >= len(power) - 2:
        start_t = obs_duration - obs_length
        stop_t = obs_duration
        peak_time = obs_duration
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
    return start_t, stop_t, peak_time


def find_best_obs_times_for_sources(
    source_names: list, all_obs_metadata: dict, beam_coverage: dict, obs_length: float = None
) -> dict:
    """Find the best observation for each source, based on the mean power level,
    then return the optimal start and stop times of an observation of obs_length
    seconds for each source.

    Parameters
    ----------
    source_names : `list`
        A list of source names.
    all_obs_metadata : `dict`
        A dictionary of metadata dictionaries.
    beam_coverage : `dict`
        A dictionary of dictionaries organised by obs IDs then source names,
        with each source entry is a list containing the enter time, the exit
        time, and the maximum zenith-normalised power reached by the source in
        the beam, and an array of powers for each time step.
    obs_length : `float`
        The desired observation length, in seconds.

    Returns
    -------
    obs_plan : `dict`
        A dictionary organised by source name, with each entry being a
        dictionary containing the obs ID, peak time, best start time, and best
        stop time, of the best observation.
    """
    # The plan will be stored as a dictionary of source names
    obs_plan = dict()

    # Loop through all specified sources
    for source in source_names:
        obsids = []
        mean_powers = []

        # Check for source beam coverage in all observations
        for obsid in all_obs_metadata:
            if source in beam_coverage[obsid]:
                _, _, _, powers, times = beam_coverage[obsid][source]
                powers = np.mean(powers, axis=1)
                obs_metadata = all_obs_metadata[obsid]

                # Find the best start/stop times for the observation
                start_t, stop_t, peak_time = plan_obs_times(
                    obs_metadata,
                    powers,
                    times,
                    obs_length=obs_length,
                )

                peak_power_segment = powers[(times > start_t) & (times < stop_t)]
                mean_powers.append(np.mean(peak_power_segment))
                obsids.append(obsid)

        if len(mean_powers) == 0:
            logger.info(f"No obs IDs found for source {source}. Omitting from download plan.")
            continue

        # Get beam coverage and metadata of best observation
        best_obsid = obsids[np.argmax(np.array(mean_powers))]
        _, _, _, powers, times = beam_coverage[best_obsid][source]
        powers = np.mean(powers, axis=1)
        obs_metadata = all_obs_metadata[best_obsid]

        # Find the best start/stop times for the best observation
        start_t, stop_t, peak_time = plan_obs_times(
            obs_metadata,
            powers,
            times,
            obs_length=obs_length,
        )

        # Store in the dictionary
        obs_plan[source] = dict(
            obsid=best_obsid,
            peak_time=peak_time,
            optimal_range=(start_t, stop_t),
        )
    return obs_plan


def plan_data_download(obs_plan: dict, savename: str = None) -> list:
    """Generate a list of downloads from an observing plan.

    Parameters
    ----------
    obs_plan : `dict`
        A dictionary organised by source name, with each entry being a
        dictionary containing the obs ID, peak time, best start time, and best
        stop time, of the best observation.
    savename : `str`, optional
        The name of the output csv file, by default None.

    Returns
    -------
    download_plans : `list`
        A list of tuples, each containing the obs ID, start time of the
        download, stop time of the download, and the sources within the
        downloaded data.
    """
    # Get a list of unique obs IDs
    all_obsids = [obs_plan[source]["obsid"] for source in obs_plan]
    unique_obsids = sorted(list(set(all_obsids)))

    # Create a dictionary of source observing times with obs IDs for keys
    source_beam_ranges = {key: [] for key in unique_obsids}
    for source in obs_plan:
        best_obsid = obs_plan[source]["obsid"]
        start_t, stop_t = obs_plan[source]["optimal_range"]
        source_beam_ranges[best_obsid].append((source, start_t, stop_t))

    download_plans = []
    for obsid in unique_obsids:
        contig_ranges = find_contiguous_ranges(source_beam_ranges[obsid], 600)
        for contig_range in contig_ranges:
            start_time, stop_time, sources = contig_range
            start_time = _round_down(start_time, 8)
            stop_time = _round_down(stop_time, 8)
            download_plans.append((obsid, start_time, stop_time, sources))

    # Write download plan to a csvfile
    if savename is not None:
        logger.info(f"Saving output file: {savename}")
        with open(savename, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["obsID", "start_time", "stop_time", "duration", "sources"])
            for download_plan in download_plans:
                obsid, start_time, stop_time, sources = download_plan
                source_info = []
                for source in sources:
                    enter_time = _round_down(source[1], 8)
                    exit_time = _round_down(source[2], 8)
                    source_info.append(f"{source[0]}@{enter_time:04.0f}-{exit_time:04.0f}")
                writer.writerow(
                    [
                        obsid,
                        f"{start_time:.0f}",
                        f"{stop_time:.0f}",
                        f"{stop_time - start_time:.0f}",
                        " ".join(source_info),
                    ]
                )
    return download_plans


def find_contiguous_ranges(sources: list, min_gap: float) -> list:
    """Combine a list of intervals into a list of contiguous intervals.

    Parameters
    ----------
    sources : `list`
        A list of tuples, each containing the source name, the start of the
        time interval, and the end of the time interval.
    min_gap : `float`
        The minimum gap between two intervals to count them as non-contiguous.
        This is to ensure that downloads are separated enough to be worth
        splitting up.

    Returns
    -------
    contig_ranges : `list`
        A list of tuples, one per contiguous interval. Each tuple contains:
        (interval_start, interval_end, [(source, enter_time, exit_time), ...])
    """
    # Sort sources by entry time
    sources.sort(key=lambda x: x[1])

    # Initialize variables
    inbeam_sources = []
    contig_ranges = []
    interval_start = None
    interval_end = None

    # Loop through each event (entry or exit)
    for isource in range(len(sources)):
        source_name, enter_time, exit_time = sources[isource]

        # For the initial interval
        if interval_start is None:
            interval_start = enter_time
        if interval_end is None:
            interval_end = exit_time

        if enter_time > interval_end + min_gap:
            # If true, then start a new interval
            contig_ranges.append((interval_start, interval_end, inbeam_sources))
            inbeam_sources = [(source_name, enter_time, exit_time)]
            interval_start = enter_time
            interval_end = exit_time
            continue

        # Otherwise, source coverage overlaps (or nearly overlaps) with the interval
        inbeam_sources.append((source_name, enter_time, exit_time))
        if exit_time > interval_end:
            interval_end = exit_time

    # For the last source, close the interval
    contig_ranges.append((interval_start, interval_end, inbeam_sources))

    return contig_ranges
