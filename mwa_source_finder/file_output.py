import logging

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mwa_source_finder import logger_setup

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


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


def plot_power_vs_time(
    pointings: list,
    obs_metadata_dict: list,
    beam_coverage: dict,
    obs_for_source: bool,
    min_power: float,
    norm: bool,
    logger: logging.Logger = None,
):
    if logger is None:
        logger = logger_setup.get_logger()

    line_combos = []
    for lsi in ["-", "--", "-.", ":"]:
        for coli in list(mcolors.TABLEAU_COLORS):
            line_combos.append([lsi, coli])

    if obs_for_source:
        # Plot of power in each observation, one plot per source
        for pointing in pointings:
            source_name = pointing["Name"]
            fig, axes = plt.subplots(ncols=2, figsize=(10, 4), dpi=300)
            ii = 0
            max_duration = 0
            for obsid in obs_metadata_dict:
                if source_name in beam_coverage[obsid]:
                    obs_duration = obs_metadata_dict[obsid]["duration"]
                    if obs_duration > max_duration:
                        max_duration = obs_duration
                    _, _, _, source_power_TF = beam_coverage[obsid][source_name]
                    source_power_FT = source_power_TF.T
                    max_idx = np.unravel_index(
                        np.argmax(source_power_FT), source_power_FT.shape
                    )[0]

                    # Plot powers
                    times_sec = np.linspace(0, obs_duration, len(source_power_FT[0]))
                    times_norm = np.linspace(0, 1, len(source_power_FT[0]))
                    axes[0].errorbar(
                        times_sec,
                        source_power_FT[max_idx],
                        ls=line_combos[ii][0],
                        c=line_combos[ii][1],
                        label=obsid,
                    )
                    axes[1].errorbar(
                        times_norm,
                        source_power_FT[max_idx],
                        ls=line_combos[ii][0],
                        c=line_combos[ii][1],
                        label=obsid,
                    )
                    ii += 1
            for ax in axes:
                ax.fill_between(
                    [0, max_duration],
                    0,
                    min_power,
                    color="grey",
                    alpha=0.3,
                    hatch="///",
                )
                if norm:
                    ax.set_ylim([0, 1])
                else:
                    ax.set_yscale("log")

            # Add legend
            axes[1].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

            # Set plot limits
            axes[0].set_xlim([0, max_duration])
            axes[1].set_xlim([0, 1])

            # Set labels
            fig.suptitle(source_name)
            axes[0].set_xlabel("Time [s]")
            axes[1].set_xlabel("Time [normalised]")
            if norm:
                axes[0].set_ylabel("Power [normalised]")
            else:
                axes[0].set_ylabel("Power")

            # Save fig
            if norm:
                plot_name = f"power_vs_time_{source_name}.png"
            else:
                plot_name = f"power_vs_time_{source_name}_norm.png"
            logger.info(f"Saving plot file: {plot_name}")
            fig.savefig(plot_name, bbox_inches="tight")
            fig.clf()

    else:
        # Plot of power for each source, one plot per observation
        for obsid in obs_metadata_dict:
            obs_duration = obs_metadata_dict[obsid]["duration"]
            fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
            for pointing in pointings:
                source_name = pointing["Name"]
                if source_name in beam_coverage[obsid]:
                    _, _, _, source_power_TF = beam_coverage[obsid][source_name]
                    source_power_FT = source_power_TF.T
                    source_power_FT = source_power_TF.T
                    max_idx = np.unravel_index(
                        np.argmax(source_power_FT), source_power_FT.shape
                    )[0]

                    # Plot powers
                    times_sec = np.linspace(0, obs_duration, len(source_power_FT[0]))
                    ax.errorbar(
                        times_sec,
                        source_power_FT[max_idx],
                        ls="-",
                    )
            ax.fill_between(
                [0, obs_duration], 0, min_power, color="grey", alpha=0.3, hatch="///"
            )

            # Set plot limits
            ax.set_ylim([0, 1])
            ax.set_xlim([0, obs_duration])

            # Set labels
            fig.suptitle(obsid)
            ax.set_ylabel("Power [normalised]")
            ax.set_xlabel("Time [s]")

            # Save fig
            fig.savefig(f"power_vs_time_{obsid}.png", bbox_inches="tight")
            fig.clf()
