import logging

import astropy.units as u
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from mwa_source_finder import beam_utils, coord_utils, logger_setup

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


def setup_axis(ax, duration, fontsize=12):
    ax.set_ylim([0, 1])
    ax.set_yticks((np.arange(11) / 10).tolist())
    ax.tick_params(axis="both", which="both", right=True, top=True)
    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=4)
    ax.tick_params(axis="both", which="minor", length=2)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.minorticks_on()
    ax.grid(ls=":", color="0.5")
    ax.set_xlim([0, duration])
    ax.set_xlabel("Time since start of observation [s]")
    ax.set_ylabel("Zenith-normalised beam power")


def plot_power_vs_time(
    source_names: list,
    all_obs_metadata: dict,
    beam_coverage: dict,
    min_power: float,
    obs_for_source: bool = False,
    logger: logging.Logger = None,
) -> None:
    """Make a plot of power vs time showing each obs ID for a source.

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
    min_power : `float`
        The minimum power to count as in the beam.
    obs_for_source : `bool`, optional
        Whether to search for observations for each source, by default False.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    line_combos = []
    for lsi in LINE_STYLES:
        for coli in list(mcolors.TABLEAU_COLORS):
            line_combos.append([lsi, coli])

    if obs_for_source:
        # Plot of power vs time for each observation, one plot per source
        for source_name in source_names:
            ii = 0
            max_duration = 0

            fig = plt.figure(figsize=(8, 4), dpi=300)
            ax = fig.add_subplot(111)

            for obsid in all_obs_metadata:
                if source_name in beam_coverage[obsid]:
                    obs_duration = all_obs_metadata[obsid]["duration"]
                    if obs_duration > max_duration:
                        max_duration = obs_duration
                    _, _, _, powers, times = beam_coverage[obsid][source_name]

                    # Plot powers
                    if ii >= len(line_combos):
                        logger.error(f"Source {source_name}: Too many obs IDs to make a power vs time plot. Skipping.")
                        return

                    ax.errorbar(
                        times,
                        powers,
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
            setup_axis(ax, max_duration)
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
            plt.close()
    else:
        # Plot of power vs time for each source, one plot per observation
        for obsid in all_obs_metadata:
            ii = 0
            max_duration = all_obs_metadata[obsid]["duration"]

            fig = plt.figure(figsize=(8, 4), dpi=300)
            ax = fig.add_subplot(111)

            for source_name in beam_coverage[obsid]:
                _, _, _, powers, times = beam_coverage[obsid][source_name]

                # Plot powers
                if ii >= len(line_combos):
                    logger.error(f"Obs ID {obsid}: Too many sources to make a power vs time plot. Skipping.")
                    return

                ax.errorbar(
                    times,
                    powers,
                    ls=line_combos[ii][0],
                    c=line_combos[ii][1],
                    label=source_name,
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
            setup_axis(ax, max_duration)
            fig.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
            fig.suptitle(f"Obs ID: {obsid}")

            # Save fig
            plot_name = f"{obsid}_power_vs_time.png"
            logger.info(f"Saving plot file: {plot_name}")
            fig.savefig(plot_name, bbox_inches="tight")
            plt.close()


def plot_beam_sky_map(
    obs_finder_result: list,
    beam_coverage: dict,
    obs_metadata: dict,
    pointings: dict,
    min_power: float,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> None:
    """Plot a figure showing the beam power and power vs time.

    Parameters
    ----------
    obs_finder_result : `list`
        The finder result for a particular obs ID. A list of lists containing
        the source name as the first entry.
    beam_coverage : `dict`
        A dictionary of dictionaries organised by obs IDs then source names,
        with each source entry is a list containing the enter time, the exit
        time, and the maximum zenith-normalised power reached by the source in
        the beam, and an array of powers for each time step.
    obs_metadata : `dict`
        A dictionary of commonly used metadata.
    pointings : `dict`
        A dictionary of dictionaries containing pointing information, organised
        by source name.
    min_power : `float`
        The minimum power to count as in the beam.
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    logger : `logging.Logger`, optional
        A custom logger to use, by default None.
    """
    if logger is None:
        logger = logger_setup.get_logger()

    az, za, powers = beam_utils.get_beam_power_sky_map(
        obs_metadata,
        norm_to_zenith=norm_to_zenith,
        logger=logger,
    )

    # Compute the timestep frames
    start_t = Time(obs_metadata["start_t"], format="gps")
    end_t = Time(obs_metadata["stop_t"], format="gps")
    source_dt = np.linspace(-3600, obs_metadata["duration"] + 3600, 50) * u.s
    obs_frame = AltAz(obstime=start_t + source_dt, location=coord_utils.TEL_LOCATION)

    # Get sky coordinates for found pulsars
    found_sources = [entry[0] for entry in obs_finder_result]
    source_names, source_RAs, source_DECs = [], [], []
    for source_name in pointings:
        pointing = pointings[source_name]
        if source_name in found_sources:
            source_names.append(source_name)
            source_RAs.append(pointing["RAJD"])
            source_DECs.append(pointing["DECJD"])

    source_coords = SkyCoord(source_RAs, source_DECs, unit=(u.deg, u.deg), frame="icrs")

    # Define the colour map
    cmap = mpl.colormaps["magma_r"]
    cmap.set_under(color="w")
    contour_levels = [0.01, 0.1, 0.5, 0.9]

    for source_name, source_radec in zip(source_names, source_coords):
        source_altaz = source_radec.transform_to(obs_frame)

        # Figure setup
        fig = plt.figure(figsize=(6, 7.5), dpi=150)
        fig.suptitle(f"Obs ID: {obs_metadata['obsid']}   Source: {source_name}", y=0.95)
        gs = mpl.gridspec.GridSpec(2, 1, hspace=0.2, height_ratios=[3, 1])

        # Polar plot
        # ----------------------------------------------------------------------
        ax_2D = plt.subplot(gs[0], projection="polar")

        # Plot the beam power
        im = ax_2D.pcolormesh(
            az,
            za,
            powers,
            vmax=1.0,
            vmin=0.01,
            rasterized=True,
            shading="auto",
            cmap=cmap,
        )

        # Plot power contours
        ax_2D.contour(az, za, powers, contour_levels, colors="k", linewidths=1, zorder=1e2)

        # Plot source paths through beam
        for altaz_step in source_altaz:
            if altaz_step.obstime < start_t:
                path_color = "lightpink"
                zorder_boost = 0
            elif altaz_step.obstime > end_t:
                path_color = "lightskyblue"
                zorder_boost = 0
            else:
                path_color = "r"
                zorder_boost = 1
            ax_2D.errorbar(
                altaz_step.az.rad,
                np.pi / 2 - altaz_step.alt.rad,
                fmt="o",
                ms=2,
                c=path_color,
                zorder=1e6 + zorder_boost,
                rasterized=True,
            )

        ax_2D.set_theta_zero_location("N")
        ax_2D.set_theta_direction(-1)
        ax_2D.grid(ls=":", color="0.5")
        ax_2D.set_yticks(np.radians([15, 35, 55, 75]))
        ax_2D.set_yticklabels([rf"${int(x)}^{{\degree}}$" for x in np.round(np.degrees(ax_2D.get_yticks()), 0)])
        ax_2D.set_xlabel("Azimuth angle [deg]", labelpad=5)
        ax_2D.set_ylabel("Zenith angle [deg]", labelpad=30)
        ax_2D.tick_params(labelsize=10)
        cbar = plt.colorbar(
            im,
            pad=0.13,
            extend="min",
            ticks=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        cbar.ax.set_ylabel("Zenith-normalised beam power", labelpad=10)
        cbar.ax.tick_params(labelsize=10)
        for contour_level in contour_levels:
            cbar.ax.axhline(contour_level, color="k", lw=1)

        # Power vs time plot
        # ----------------------------------------------------------------------
        ax_1D = plt.subplot(gs[1])
        _, _, _, source_power, _ = beam_coverage[obs_metadata["obsid"]][source_name]

        power_dt = np.linspace(0, obs_metadata["duration"], len(source_power))
        ax_1D.errorbar(
            power_dt,
            source_power,
            fmt="k-",
        )

        ax_1D.fill_between(
            [0, obs_metadata["duration"]],
            0,
            min_power,
            color="grey",
            alpha=0.2,
            hatch="///",
        )

        ax_1D.set_xlabel("Time since start of observation [s]")
        ax_1D.set_ylabel("Z.N. beam power")
        ax_1D.set_ylim([0, 1])
        ax_1D.set_xlim([0, obs_metadata["duration"]])
        ax_1D.set_yticks([0, 0.1, 0.5, 0.9, 1.0])
        ax_1D.grid(ls=":", color="0.5")
        ax_1D.tick_params(labelsize=10)

        fig_name = f"{obs_metadata['obsid']}_{source_name}_sky_beam_power.png"
        logger.info(f"Saving plot file: {fig_name}")
        plt.savefig(fig_name, bbox_inches="tight")
        plt.close()
