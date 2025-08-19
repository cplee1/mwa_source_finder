import logging

import astropy.units as u
import cmasher as cmr
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from .beam import get_beam_power_sky_map
from .constants import LINE_STYLES, TEL_LOCATION

__all__ = [
    "setup_axis",
    "plot_power_vs_time",
    "plot_beam_sky_map",
    "plot_multisource_beam_sky_map",
]

logger = logging.getLogger(__name__)


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
    ax.set_xlabel("Elapsed Time [s]")
    ax.set_ylabel("Zenith-Normalised Beam Power")


def get_source_path(
    start_t_abs: Time,
    start_t_rel: float,
    end_t_rel: float,
    source_radec: AltAz,
    num_points: int = 50,
) -> np.ndarray:
    """Compute the path of a source in Alt/ZA coordinates.

    Parameters
    ----------
    start_t_abs : `astropy.time.Time`
        A Time object defining the absolute start time of the observation.
    start_t_rel : `float`
        The start second of the path, relative to the start second of the
        observation.
    end_t_rel : `float`
        The end second of the path, relative to the start second of the
        observation.
    source_radec : `astropy.coordinates.SkyCoord`
        A SkyCoord object defining the RA/Dec coordinates of the source.
    num_points : `int`, optional
        The number of points in the path, by default 50.

    Returns
    -------
    altaz_path : `np.ndarray`
        A (2, num_points) array containing the Alt/ZA coordinates of the source
        in radians.
    """
    source_dt = np.linspace(start_t_rel, end_t_rel, num_points) * u.s
    obs_frame = AltAz(obstime=start_t_abs + source_dt, location=TEL_LOCATION)
    source_altaz = source_radec.transform_to(obs_frame)
    altaz_path = np.empty((2, len(source_altaz)))
    for ii, altaz_step in enumerate(source_altaz):
        altaz_path[0, ii] = altaz_step.az.rad
        altaz_path[1, ii] = np.pi / 2 - altaz_step.alt.rad
    return altaz_path


def plot_power_vs_time(
    source_names: list,
    all_obs_metadata: dict,
    beam_coverage: dict,
    min_power: float,
    obs_for_source: bool = False,
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
    """
    line_combos = []
    for lsi in LINE_STYLES:
        for coli in list(mcolors.TABLEAU_COLORS):
            line_combos.append([lsi, coli])

    if obs_for_source:
        # Plot of power vs time for each observation, one plot per source
        for source_name in source_names:
            max_duration = 0

            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)

            for ii, obsid in enumerate(all_obs_metadata):
                if source_name in beam_coverage[obsid]:
                    obs_duration = all_obs_metadata[obsid]["duration"]
                    if obs_duration > max_duration:
                        max_duration = obs_duration
                    _, _, _, powers, times = beam_coverage[obsid][source_name]

                    # Plot powers
                    if ii >= len(line_combos):
                        logger.error(
                            f"Source {source_name}: Too many obs IDs to make a "
                            + "power vs time plot. Skipping."
                        )
                        return

                    for ifreq in range(powers.shape[1]):
                        label = None
                        if ifreq == 0:
                            label = obsid
                        ax.errorbar(
                            times,
                            powers[:, ifreq],
                            ls=line_combos[ii][0],
                            c=line_combos[ii][1],
                            label=label,
                        )

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
                ncol=3,
            )
            fig.suptitle(f"Source: {source_name.replace('-', '$-$')}")

            # Save fig
            plot_name = f"{source_name}_power_vs_time.png"
            logger.info(f"Saving plot file: {plot_name}")
            fig.savefig(plot_name, bbox_inches="tight")
            plt.close()
    else:
        # Plot of power vs time for each source, one plot per observation
        for obsid in all_obs_metadata:
            max_duration = all_obs_metadata[obsid]["duration"]

            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)

            for ii, source_name in enumerate(beam_coverage[obsid]):
                _, _, _, powers, times = beam_coverage[obsid][source_name]

                # Plot powers
                if ii >= len(line_combos):
                    logger.error(
                        f"Obs ID {obsid}: Too many sources to make a "
                        + "power vs time plot. Skipping."
                    )
                    return

                for ifreq in range(powers.shape[1]):
                    label = None
                    if ifreq == 0:
                        label = source_name
                    ax.errorbar(
                        times,
                        powers[:, ifreq],
                        ls=line_combos[ii][0],
                        c=line_combos[ii][1],
                        label=label,
                    )

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
    min_power: float = None,
    norm_to_zenith: bool = True,
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
    min_power : `float`, optional
        If specified, will plot the minimum power on the power vs time plot.
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    """
    az, za, powers = get_beam_power_sky_map(obs_metadata, norm_to_zenith=norm_to_zenith)

    # Compute the timestep frames
    start_t = Time(obs_metadata["start_t"], format="gps")

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
    cmap = plt.get_cmap("cmr.arctic_r")
    cmap.set_under(color="w")
    contour_levels = [0.01, 0.1, 0.5, 0.9]

    for source_name, source_radec in zip(source_names, source_coords, strict=True):
        path0 = get_source_path(start_t, -3600, 0, source_radec)
        path1 = get_source_path(start_t, 0, obs_metadata["duration"], source_radec)
        path2 = get_source_path(
            start_t,
            obs_metadata["duration"],
            obs_metadata["duration"] + 3600,
            source_radec,
        )

        # Figure setup
        fig = plt.figure(figsize=(5, 6.5))
        fig.suptitle(
            f"Obs ID: {obs_metadata['obsid']}   "
            + f"Source: {source_name.replace('-', '$-$')}",
            y=1.02,
        )
        gs = mpl.gridspec.GridSpec(2, 1, hspace=0.2, height_ratios=[2.5, 1])

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
        ax_2D.contour(
            az, za, powers, contour_levels, colors="k", linewidths=1, zorder=1e2
        )

        # Plot source paths through beam
        for path, ls, col, lab in zip(
            [path0, path1, path2],
            ["-", "-", "-"],
            ["lightpink", "r", "lightskyblue"],
            ["1 h before", "Observation", "1 h after"],
            strict=True,
        ):
            ax_2D.errorbar(
                path[0, :],
                path[1, :],
                ls=ls,
                lw=2,
                c=col,
                zorder=1e6,
                rasterized=True,
                label=lab,
            )
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.48, 0.975),
            fancybox=True,
            ncol=3,
        )

        ax_2D.set_theta_zero_location("N")
        ax_2D.set_theta_direction(-1)
        ax_2D.set_rlabel_position(157.5)
        ax_2D.grid(ls=":", color="0.5")
        ax_2D.set_yticks(np.radians([15, 35, 55, 75]))
        ax_2D.set_yticklabels(
            [
                "${}^\\circ$".format(int(x))
                for x in np.round(np.degrees(ax_2D.get_yticks()), 0)
            ]
        )
        ax_2D.set_xlabel("Azimuth Angle [deg]", labelpad=5)
        ax_2D.set_ylabel("Zenith Angle [deg]", labelpad=30)
        ax_2D.tick_params(labelsize=10)
        cbar = plt.colorbar(
            im,
            pad=0.13,
            extend="min",
            ticks=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        cbar.ax.set_ylabel("Zenith-Normalised Beam Power", labelpad=10)
        cbar.ax.tick_params(labelsize=10)
        for contour_level in contour_levels:
            cbar.ax.axhline(contour_level, color="k", lw=1)

        # Power vs time plot
        # ----------------------------------------------------------------------
        ax_1D = plt.subplot(gs[1])
        _, _, _, source_powers, times = beam_coverage[obs_metadata["obsid"]][
            source_name
        ]
        source_power = np.mean(source_powers, axis=1)

        ax_1D.errorbar(
            times,
            source_power,
            fmt="k-",
            lw=1,
        )

        if min_power is not None:
            ax_1D.fill_between(
                [0, obs_metadata["duration"]],
                0,
                min_power,
                color="grey",
                alpha=0.2,
                hatch="///",
            )

        ax_1D.set_xlabel("Elapsed Time [s]")
        ax_1D.set_ylabel("Z.N. Beam Power")
        ax_1D.set_ylim([0, 1])
        ax_1D.set_xlim([0, obs_metadata["duration"]])
        ax_1D.set_yticks([0, 0.1, 0.5, 0.9, 1.0])
        ax_1D.grid(ls=":", color="0.5")
        ax_1D.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax_1D.tick_params(
            axis="both",
            which="both",
            right=True,
            top=True,
            direction="in",
            labelsize=10,
        )

        fig_name = f"{obs_metadata['obsid']}_{source_name}_sky_beam_power.png"
        logger.info(f"Saving plot file: {fig_name}")
        plt.savefig(fig_name, bbox_inches="tight")
        plt.close()


def plot_multisource_beam_sky_map(
    obs_finder_result: list,
    beam_coverage: dict,
    obs_metadata: dict,
    pointings: dict,
    min_power: float = None,
    norm_to_zenith: bool = True,
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
    min_power : `float`, optional
        If specified, will plot the minimum power on the power vs time plot.
    norm_to_zenith : `bool`, optional
        Whether to normalise the powers to zenith, by default True.
    """
    az, za, powers = get_beam_power_sky_map(obs_metadata, norm_to_zenith=norm_to_zenith)

    # Compute the timestep frames
    start_t = Time(obs_metadata["start_t"], format="gps")

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
    cmap = plt.get_cmap("cmr.arctic_r")
    cmap.set_under(color="w")
    contour_levels = [0.01, 0.1, 0.5, 0.9]

    # Figure setup
    fig = plt.figure(figsize=(5, 6.5))
    gs0 = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[2.5, 1])
    gs00 = gs0[0]
    gs01 = gs0[1]

    ax_2D = fig.add_subplot(gs00, projection="polar")
    ax_1D = fig.add_subplot(gs01)

    # Polar plot
    # ----------------------------------------------------------------------
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

    for ii, source_radec in enumerate(source_coords):
        path0 = get_source_path(start_t, -3600, 0, source_radec)
        path1 = get_source_path(start_t, 0, obs_metadata["duration"], source_radec)
        path2 = get_source_path(
            start_t,
            obs_metadata["duration"],
            obs_metadata["duration"] + 3600,
            source_radec,
        )

        # Plot source paths through beam
        for path, ls, col, lab, alpha in zip(
            [path0, path1, path2],
            ["-", "-", "-"],
            ["tab:red", "tab:red", "tab:red"],
            ["1 h before", "Observation", "1 h after"],
            [0.4, 1, 0.4],
            strict=True,
        ):
            ax_2D.errorbar(
                path[0, :],
                path[1, :],
                ls=ls,
                lw=1.7,
                c=col,
                alpha=alpha,
                zorder=1e6,
                rasterized=True,
                label=lab if ii == 0 else None,
            )

    ax_2D.set_theta_zero_location("N")
    ax_2D.set_theta_direction(-1)
    ax_2D.set_rlabel_position(157.5)
    ax_2D.grid(ls=":", color="0.5")
    ax_2D.set_ylim(np.radians([0, 75]))
    ax_2D.set_yticks(np.radians([15, 30, 45, 60]))
    ax_2D.set_yticklabels(
        [
            "${}^\\circ$".format(int(x))
            for x in np.round(np.degrees(ax_2D.get_yticks()), 0)
        ]
    )
    ax_2D.set_xlabel("Azimuth Angle [deg]", labelpad=5)
    ax_2D.set_ylabel("Zenith Angle [deg]", labelpad=30)
    ax_2D.tick_params(axis="both", which="both", direction="in", labelsize=10)
    cbar = plt.colorbar(
        im,
        pad=0.13,
        extend="min",
        ticks=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    cbar.ax.set_ylabel("Zenith-Normalised Beam Power", labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    for contour_level in contour_levels:
        cbar.ax.axhline(contour_level, color="k", lw=1)

    # Power vs time plot
    # ----------------------------------------------------------------------
    lstyles = ["-", "--", "-.", ":"]
    for ii, source_name in enumerate(source_names):
        _, _, _, source_power, _ = beam_coverage[obs_metadata["obsid"]][source_name]
        source_power = np.mean(source_power, axis=1)
        power_dt = np.linspace(0, obs_metadata["duration"], len(source_power))

        ax_1D.errorbar(
            power_dt,
            source_power,
            marker="none",
            ls=lstyles[ii % 4],
            c="k",
            lw=1,
        )

    if min_power is not None:
        ax_1D.fill_between(
            [0, obs_metadata["duration"]],
            0,
            min_power,
            color="grey",
            alpha=0.2,
            hatch="///",
        )

    ax_1D.set_xlabel("Elapsed Time [s]")
    ax_1D.set_ylabel("Z.N. Beam Power")
    ax_1D.set_ylim([0, 1])
    ax_1D.set_xlim([0, obs_metadata["duration"]])
    ax_1D.minorticks_on()
    ax_1D.set_yticks([0, 0.1, 0.5, 0.9, 1.0])
    ax_1D.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax_1D.tick_params(
        axis="both", which="both", right=True, top=True, direction="in", labelsize=10
    )
    ax_1D.grid(ls=":", color="0.5")

    fig_name = f"{obs_metadata['obsid']}_multisource_sky_beam_power"
    logger.info(f"Saving plot file: {fig_name}.png")
    plt.savefig(fig_name + ".png", bbox_inches="tight")

    plt.close()
