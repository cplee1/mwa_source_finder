import logging

import numpy as np
from astropy.table import Table

from mwa_source_finder import logger_setup


def write_output_source_files(
    finder_result: dict,
    obs_metadata_dict: dict,
    min_z_power: float,
    logger: logging.Logger = None,
):
    """Write finder results for each source.

    Parameters
    ----------
    finder_result : dict
        A dictionary containing the results organised by source.
    obs_metadata_dict : dict
        A dictionary of observation metadata dictionaries.
    min_z_power: float
        The minimum zenith-normalised power.
    logger : logging.Logger, optional
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
        data["Enter"].format = "%.2f"
        data["Exit"].format = "%.2f"
        data["Power"].format = "%.2f"
        data["Freq"].format = "%.2f"
        data["BW"].format = "%.2f"

        out_file = f"{source}_obsIDs.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        divider_str = "# " + "-" * 78 + "\n"
        header = (
            divider_str
            + f"# All observations where the source reaches a zenith normalised power >{min_z_power}\n"
            + divider_str
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
    min_z_power: float,
    logger: logging.Logger = None,
):
    """Write finder results for each observation.

    Parameters
    ----------
    finder_result : dict
        A dictionary containing the results organised by observation ID.
    obs_metadata_dict : dict
        A dictionary of observation metadata dictionaries.
    logger : logging.Logger, optional
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
        data["Enter"].format = "%.2f"
        data["Exit"].format = "%.2f"
        data["Power"].format = "%.2f"

        out_file = f"{obsid}_sources.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        obs_metadata = obs_metadata_dict[obsid]
        divider_str = "# " + "-" * 78 + "\n"
        header = (
            divider_str
            + f"# All sources which reach a zenith normalised power >{min_z_power}\n"
            + f"# Obs ID      -- {obsid}\n"
            + f"# Duration    -- {obs_metadata['duration']} s\n"
            + f"# Centre freq -- {obs_metadata['centrefreq']} MHz\n"
            + f"# Bandwidth   -- {obs_metadata['bandwidth']} MHz\n"
            + divider_str
            + "# Column headers:\n"
            + "# Obs ID -- Observation ID\n"
            + "# Enter  -- The fraction of the observation when the source enters the beam\n"
            + "# Exit   -- The fraction of the observation when the source exits the beam\n"
            + "# Power  -- The maximum power of the source in the beam\n"
            + divider_str
            + "\n"
        )

        with open(out_file, "r") as f:
            lines = f.readlines()
        lines.insert(0, header)
        with open(out_file, "w") as f:
            f.writelines(lines)
