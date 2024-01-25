#TODO: Prepend descriptions to header

import numpy as np
from astropy.table import Table

from mwa_source_finder import logger_setup


def write_output_source_files(finder_result, logger=None):
    if logger is None:
        logger = logger_setup.get_logger()

    for source in finder_result:
        if len(finder_result[source]) == 0:
            continue
        source_data = np.array(finder_result[source])
        data = Table(
            names=["Obs ID", "Enter", "Exit", "Power", "Dur", "Freq", "BW"],
            dtype=[int, float, float, float, int, float, float],
            descriptions=[
                "Observation ID",
                "The fraction of the observation when the source enters the beam",
                "The fraction of the observation when the source exits the beam",
                "The maximum power of the source",
                "The length of the observation",
                "The centre frequency of the observation",
                "The bandwidth of the observation",
            ],
            rows=source_data,
        )
        data["Enter"].format = "%.2f"
        data["Exit"].format = "%.2f"
        data["Power"].format = "%.2f"
        data["Freq"].format = "%.2f"
        data["BW"].format = "%.2f"
        out_file = f"{source}_obsIDs.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)


def write_output_obs_files(finder_result, logger=None):
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
