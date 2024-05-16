import logging

import numpy as np
from astropy.table import Table

from mwa_source_finder import logger_setup


def write_output_source_files(
    finder_result: dict,
    obs_metadata_dict: dict,
    freq_mode: str,
    norm_mode: str,
    min_power: float,
    obs_plan: dict = None,
    logger: logging.Logger = None,
) -> None:
    """Write finder results for each source.

    Parameters
    ----------
    finder_result : `dict`
        A dictionary containing the results organised by source.
    obs_metadata_dict : `dict`
        A dictionary of observation metadata dictionaries.
    freq_mode : `str`
        The frequency mode used ['low', 'centre', 'high'].
    norm_mode : `str`
        The beam normalisation mode used ['zenith', 'beam'].
    min_power : `float`, optional
        The minimum power to count as in the beam.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    for source in finder_result:
        if len(finder_result[source]) == 0:
            continue
        beam_data = finder_result[source]
        beam_data_extended = []
        for row in beam_data:
            obs_metadata = obs_metadata_dict[row[0]]
            row += [obs_metadata["duration"]]
            row += [obs_metadata["centrefreq"]]
            row += [obs_metadata["bandwidth"]]
            beam_data_extended.append(row)
        data = Table(
            names=["Obs ID", "Enter", "Exit", "Power", "Dur", "Freq", "BW"],
            dtype=[int, float, float, float, int, float, float],
            rows=beam_data_extended,
        )
        data["Enter"].format = "%.3f"
        data["Exit"].format = "%.3f"
        data["Power"].format = "%.3f"
        data["Freq"].format = "%.2f"
        data["BW"].format = "%.2f"

        out_file = f"{source}_obsIDs.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        divider_str = "# " + "-" * 78 + "\n"
        if obs_plan is not None:
            source_obs_plan = obs_plan[source]
            best_obsid = source_obs_plan["obsid"]
            start_t, stop_t = source_obs_plan["optimal_range"]
            gps_start_t = obs_metadata_dict[best_obsid]["start_t"]
            
            obs_plan_str = (
                "# Observation plan:\n"
                + f"# Best obs          -- {best_obsid}\n"
                + f"# Start GPS time    -- {gps_start_t+start_t:.0f} s\n"
                + f"# Stop GPS time     -- {gps_start_t+stop_t:.0f} s\n"
                + f"# Start time offset -- {start_t:.0f} s\n"
                + f"# Stop time offset  -- {stop_t:.0f} s\n"
                + divider_str
            )
        else:
            obs_plan_str = ""
        header = (
            divider_str
            + "# Source finder settings:\n"
            + f"# Freq mode      -- {freq_mode}\n"
            + f"# Norm mode      -- {norm_mode}\n"
            + f"# Min norm power -- {min_power:.2f}\n"
            + divider_str
            + obs_plan_str
            + "# Column headers:\n"
            + "# Obs ID -- Observation ID\n"
            + "# Enter  -- The fraction of the observation when the source enters the beam\n"
            + "# Exit   -- The fraction of the observation when the source exits the beam\n"
            + "# Power  -- The maximum power of the source in the beam\n"
            + "# Dur    -- The length of the observation in seconds\n"
            + "# Freq   -- The centre frequency of the observation in MHz\n"
            + "# BW     -- The bandwidth of the observation in MHz\n"
            + divider_str
            + "\n"
        )

        with open(out_file, "r") as f:
            lines = f.readlines()
        lines.insert(0, header)
        with open(out_file, "w") as f:
            f.writelines(lines)


def write_output_obs_files(
    finder_result: dict,
    obs_metadata_dict: dict,
    t_start: float,
    t_end: float,
    norm_mode: str,
    min_power: float,
    logger: logging.Logger = None,
) -> None:
    """Write finder results for each observation.

    Parameters
    ----------
    finder_result : `dict`
        A dictionary containing the results organised by observation ID.
    obs_metadata_dict : `dict`
        A dictionary of observation metadata dictionaries.
    t_start : `float`
        Start time of the search, as a fraction of the full observation.
    t_end : `float`
        End time of the search, as a fraction of the full observation.
    norm_mode : `str`
        The beam normalisation mode used ['zenith', 'beam'].
    min_power : `float`, optional
        The minimum power to count as in the beam.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    for obsid in finder_result:
        if len(finder_result[obsid]) == 0:
            continue
        obs_data = np.array(finder_result[obsid])
        data = Table(
            names=["Name", "Enter", "Exit", "Power"],
            dtype=[str, float, float, float],
            descriptions=[
                "Source name",
                "The fraction of the observation when the source enters the beam",
                "The fraction of the observation when the source exits the beam",
                "The maximum power of the source",
            ],
            rows=obs_data,
        )
        data["Enter"].format = "%.3f"
        data["Exit"].format = "%.3f"
        data["Power"].format = "%.3f"

        out_file = f"{obsid}_sources.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        obs_metadata = obs_metadata_dict[obsid]
        t_start_offset = t_start*obs_metadata['duration']
        t_stop_offset = t_end*obs_metadata['duration']
        divider_str = "# " + "-" * 78 + "\n"
        header = (
            divider_str
            + "# Observation metadata:\n"
            + f"# Obs ID      -- {obsid}\n"
            + f"# Centre freq -- {obs_metadata['centrefreq']:.2f} MHz\n"
            + f"# Bandwidth   -- {obs_metadata['bandwidth']:.2f} MHz\n"
            + f"# Duration    -- {obs_metadata['duration']:.0f} s\n"
            + divider_str
            + "# Source finder settings:\n"
            + f"# Start GPS time    -- {t_start_offset+obs_metadata['start_t']:.0f} s\n"
            + f"# Stop GPS time     -- {t_stop_offset+obs_metadata['start_t']:.0f} s\n"
            + f"# Start time offset -- {t_start_offset:.0f} s\n"
            + f"# Stop time offset  -- {t_stop_offset:.0f} s\n"
            + f"# Frequency         -- {obs_metadata['evalfreq']/1e6:.2f} MHz\n"
            + f"# Beam norm         -- {norm_mode}\n"
            + f"# Min norm power    -- {min_power:.2f}\n"
            + divider_str
            + "# Column headers:\n"
            + "# Name  -- Source name\n"
            + "# Enter -- The fraction of the time range when the source enters the beam\n"
            + "# Exit  -- The fraction of the time range when the source exits the beam\n"
            + "# Power -- The maximum beam power towards the source within the time range\n"
            + divider_str
            + "\n"
        )

        with open(out_file, "r") as f:
            lines = f.readlines()
        lines.insert(0, header)
        with open(out_file, "w") as f:
            f.writelines(lines)


def invert_finder_results(finder_results: dict, obs_for_source: bool = True) -> dict:
    """Invert the finder_results dictonary so that the heirarchy of obs IDs and
    source names are swapped.

    Parameters
    ----------
    finder_results : `dict`
        A dictionary containing the finder results.
    obs_for_source : `bool`, optional
        If true, then the assumed heirarchy is (source, obsid), by default True.

    Returns
    -------
    new_finder_results : `dict`
        A dictionary containing the inverted finder results.
    """
    new_finder_results = dict()

    if obs_for_source:
        for source in finder_results:
            finder_result = finder_results[source]
            for obsid_data in finder_result:
                if len(obsid_data) == 4:
                    obsid, enter_beam, exit_beam, max_power = obsid_data
                else:
                    obsid, enter_beam, exit_beam, max_power, dur, fctr, bw = obsid_data

                if obsid not in new_finder_results:
                    new_finder_results[obsid] = []

                if len(obsid_data) == 4:
                    new_finder_results[obsid].append(
                        [source, enter_beam, exit_beam, max_power]
                    )
                else:
                    new_finder_results[obsid].append(
                        [source, enter_beam, exit_beam, max_power, dur, fctr, bw]
                    )
    else:
        for obsid in finder_results:
            finder_result = finder_results[obsid]
            for source_data in finder_result:
                source, enter_beam, exit_beam, max_power = source_data
                if source not in new_finder_results:
                    new_finder_results[source] = []
                new_finder_results[source].append(
                    [obsid, enter_beam, exit_beam, max_power]
                )

    return new_finder_results
