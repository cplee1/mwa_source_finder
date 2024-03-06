import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mwa_hyperbeam
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u

from mwa_source_finder import logger_setup, coord_utils

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


def get_beam_power_grid(
    obs_metadata: dict,
    az: np.ndarray,
    za: np.ndarray,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
) -> np.ndarray:
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

    # Define the sky matrix
    S = np.eye(2) / 2

    logger.debug(f"Obs ID {obs_metadata['obsid']}: Computing jones matrix")
    jones = beam.calc_jones_array(
        az.flatten(),
        za.flatten(),
        obs_metadata["evalfreq"],
        obs_metadata["delays"],
        np.ones_like(obs_metadata["delays"]),
        norm_to_zenith,
    )

    logger.debug(f"Obs ID {obs_metadata['obsid']}: Computing beam power")
    J = jones.reshape(-1, 2, 2)
    K = np.conjugate(J).T
    powers = np.einsum("Nki,ij,jkN->N", J, S, K, optimize=True).real
    powers_grid = powers.reshape(az.shape)
    return powers_grid


def make_grid(az0, az1, za0, za1, n):
    _az = np.linspace(az0, az1, n)
    _za = np.linspace(za0, za1, n)
    az, za = np.meshgrid(_az, _za)

    return az, za


def generate_beam_sky_map(
    finder_result: list,
    beam_coverage: dict,
    obs_metadata: dict,
    pointings: dict,
    min_power: float,
    norm_to_zenith: bool = True,
    logger: logging.Logger = None,
):
    az, za = make_grid(0, 2 * np.pi, 0, 0.95 * np.pi / 2, 600)

    powers_grid = get_beam_power_grid(
        obs_metadata,
        az,
        za,
        norm_to_zenith=norm_to_zenith,
        logger=logger,
    )

    # Compute the timestep frames
    start_t = Time(obs_metadata["start_t"], format="gps")
    end_t = Time(obs_metadata["stop_t"], format="gps")
    source_dt = np.linspace(-3600, obs_metadata["duration"] + 3600, 50) * u.s
    obs_frame = AltAz(obstime=start_t + source_dt, location=coord_utils.TEL_LOCATION)

    # Get sky coordinates for found pulsars
    found_sources = [entry[0] for entry in finder_result]
    source_names, source_RAs, source_DECs = [], [], []
    for pointing in pointings:
        if pointing["name"] in found_sources:
            source_names.append(pointing["name"])
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
            powers_grid,
            vmax=1.0,
            vmin=0.01,
            rasterized=True,
            shading="auto",
            cmap=cmap,
        )

        # Plot power contours
        ax_2D.contour(
            az, za, powers_grid, contour_levels, colors="k", linewidths=1, zorder=1e2
        )

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
        ax_2D.set_yticklabels(
            [
                rf"${int(x)}^{{\degree}}$"
                for x in np.round(np.degrees(ax_2D.get_yticks()), 0)
            ]
        )
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
        _, _, _, source_power = beam_coverage[obs_metadata["obsid"]][source_name]

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
