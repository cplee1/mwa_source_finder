import logging
import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
import mwa_hyperbeam

from mwa_source_finder import logger_setup, coord_utils

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


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
        obs_metadata_dict[obsid]["evalfreq"] = freq

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
                beam_coverage[obsid][pointing["name"]] = [
                    beam_enter,
                    beam_exit,
                    np.amax(source_obs_power),
                    source_obs_power,
                ]
        if not beam_coverage[obsid]:
            beam_coverage.pop(obsid)
            obs_metadata_dict.pop(obsid)
    return beam_coverage, obs_metadata_dict


def setup_ticks(ax, fontsize=12):
    ax.tick_params(axis="both", which="both", right=True, top=True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=4)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.minorticks_on()


def plot_power_vs_time(
    source_names: list,
    obs_metadata_dict: list,
    beam_coverage: dict,
    min_power: float,
    logger: logging.Logger = None,
):
    """Make a plot of power vs time showing each obs ID for a source.

    Parameters
    ----------
    source_names : list
        A list of pointing dictionaries.
    obs_metadata_dict : list
        A dictionary of metadata dictionaries.
    beam_coverage : dict
        A dictionary organised by obs IDs then source names, with each source
        having a list the following:

            enter_beam : float
                The fraction of the observation where the source enters the beam.
            exit_beam : float
                The fraction of the observation where the source exits the beam.
            max_pow: float
                The maximum power reached within the beam.

    min_power : float
        The minimum power to count as in the beam.
    logger : logging.Logger, optional
        A custom logger to use, by default None.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    line_combos = []
    for lsi in [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1, 1, 1, 1, 1)),
        (0, (5, 1, 5, 1, 1, 1)),
        (0, (5, 1, 5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 5, 1, 5, 1, 1, 1)),
        (0, (5, 1, 5, 1, 5, 1, 1, 1, 1, 1)),
        (0, (5, 1, 1, 1, 5, 1, 5, 1, 1, 1)),
        (0, (5, 1, 1, 1, 5, 1, 1, 1, 1, 1)),
    ]:
        for coli in list(mcolors.TABLEAU_COLORS):
            line_combos.append([lsi, coli])

    # Plot of power in each observation, one plot per source
    for source_name in source_names:
        ii = 0
        max_duration = 0

        fig = plt.figure(figsize=(8, 4), dpi=300)
        ax = fig.add_subplot(111)

        for obsid in obs_metadata_dict:
            if source_name in beam_coverage[obsid]:
                obs_duration = obs_metadata_dict[obsid]["duration"]
                if obs_duration > max_duration:
                    max_duration = obs_duration
                _, _, _, source_power = beam_coverage[obsid][source_name]

                # Plot powers
                times_sec = np.linspace(0, obs_duration, len(source_power))
                ax.errorbar(
                    times_sec,
                    source_power,
                    ls=line_combos[ii][0],
                    c=line_combos[ii][1],
                    label=obsid,
                )
                ii += 1
        
        ax.fill_between(
            [0, max_duration],
            0,
            min_power,
            color="grey",
            alpha=0.2,
            hatch="///",
        )
        ax.set_ylim([0, 1])
        ax.set_yticks((np.arange(11)/10).tolist())
        setup_ticks(ax)
        ax.grid(ls=":", color="0.5")
        ax.set_xlim([0, max_duration])
        ax.set_xlabel("Time since start of observation [s]")
        ax.set_ylabel("Zenith-normalised beam power")
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        fig.suptitle(f"Source: {source_name}")

        # Save fig
        plot_name = f"{source_name}_power_vs_time.png"
        logger.info(f"Saving plot file: {plot_name}")
        fig.savefig(plot_name, bbox_inches="tight")
        fig.clf()
