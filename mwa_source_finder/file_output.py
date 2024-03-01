import logging

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec

from mwa_source_finder import logger_setup

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


def write_output_source_files(
    finder_result: dict,
    obs_metadata_dict: dict,
    freq_mode: str,
    norm_mode: str,
    min_power: float,
    logger: logging.Logger = None,
):
    """Write finder results for each source.

    Parameters
    ----------
    finder_result : dict
        A dictionary containing the results organised by source.
    obs_metadata_dict : dict
        A dictionary of observation metadata dictionaries.
    freq_mode: str
        The frequency mode used ['low', 'centre', 'high'].
    norm_mode: str
        The beam normalisation mode used ['zenith', 'beam'].
    min_power : float, optional
        The minimum power to count as in the beam.
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
        data["Enter"].format = "%.3f"
        data["Exit"].format = "%.3f"
        data["Power"].format = "%.3f"
        data["Freq"].format = "%.2f"
        data["BW"].format = "%.2f"

        out_file = f"{source}_obsIDs.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        divider_str = "# " + "-" * 78 + "\n"
        header = (
            divider_str
            + "# Source finder settings:\n"
            + f"# Freq. mode -- {freq_mode}\n"
            + f"# Norm. mode -- {norm_mode}\n"
            + f"# Min power  -- {min_power:.2f}\n"
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
    t_start: float,
    t_end: float,
    freq: float,
    norm_mode: str,
    min_power: float,
    logger: logging.Logger = None,
):
    """Write finder results for each observation.

    Parameters
    ----------
    finder_result : dict
        A dictionary containing the results organised by observation ID.
    obs_metadata_dict : dict
        A dictionary of observation metadata dictionaries.
    t_start: float
        Start time of the search, as a fraction of the full observation.
    t_end: float
        End time of the search, as a fraction of the full observation.
    freq: float
        The frequency of the search.
    norm_mode: str
        The beam normalisation mode used ['zenith', 'beam'].
    min_power : float, optional
        The minimum power to count as in the beam.
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
        data["Enter"].format = "%.3f"
        data["Exit"].format = "%.3f"
        data["Power"].format = "%.3f"

        out_file = f"{obsid}_sources.txt"
        logger.info(f"Saving output file: {out_file}")
        data.write(out_file, format="ascii.fixed_width_two_line", overwrite=True)

        obs_metadata = obs_metadata_dict[obsid]
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
            + f"# Start time -- {t_start*obs_metadata['duration']:.2f} s\n"
            + f"# End time   -- {t_end*obs_metadata['duration']:.2f} s\n"
            + f"# Frequency  -- {freq/1e6:.2f} MHz\n"
            + f"# Beam norm  -- {norm_mode}\n"
            + f"# Min power  -- {min_power:.2f}\n"
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
        gs = gridspec.GridSpec(1, 2, wspace=0.1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        axes = (ax0, ax1)

        for obsid in obs_metadata_dict:
            if source_name in beam_coverage[obsid]:
                obs_duration = obs_metadata_dict[obsid]["duration"]
                if obs_duration > max_duration:
                    max_duration = obs_duration
                _, _, _, source_power = beam_coverage[obsid][source_name]
                
                # Plot powers
                times_sec = np.linspace(0, obs_duration, len(source_power))
                times_norm = np.linspace(0, 1, len(source_power))
                axes[0].errorbar(
                    times_sec,
                    source_power,
                    ls=line_combos[ii][0],
                    c=line_combos[ii][1],
                    label=obsid,
                )
                axes[1].errorbar(
                    times_norm,
                    source_power,
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
            ax.set_ylim([0, 1])
            setup_ticks(ax)
        axes[0].set_xlim([0, max_duration])
        axes[1].set_xlim([0, 1])
        axes[1].set_yticklabels([])
        axes[0].set_xlabel("Time [s]")
        axes[1].set_xlabel("Time [normalised]")
        axes[0].set_ylabel("Power [normalised]")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        fig.suptitle(source_name)

        # Save fig
        plot_name = f"power_vs_time_{source_name}.png"
        logger.info(f"Saving plot file: {plot_name}")
        fig.savefig(plot_name, bbox_inches="tight")
        fig.clf()
