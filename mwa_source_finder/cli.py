import sys
import argparse

import numpy as np

from mwa_source_finder import logger_setup, finder, file_output


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Code to find sources in MWA VCS observations.",
        add_help=False,
    )
    loglevels = logger_setup.get_log_levels()

    # Program arguments
    optional = parser.add_argument_group("Program arguments")
    optional.add_argument(
        "-h", "--help", action="help", help="Show this help information and exit."
    )
    optional.add_argument(
        "-L",
        "--loglvl",
        type=str,
        choices=loglevels,
        default="INFO",
        help="Logger verbosity level.",
    )

    # Source arguments
    source_args = parser.add_argument_group(
        "Source arguments",
        "Options to specify the source(s) to find. The default is all pulsars.",
    )
    source_args.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="*",
        default=None,
        help="A list of sources to find. Sources can be specified as pulsar names "
        + "or equatorial coordinates separated by an underscore. Coordinates can "
        + "be in either decimal or sexigesimal format. For example, the following "
        + 'arguments are all valid: "B1451-68" "J1456-6843" "14:55:59.92_-68:43:39.50" '
        + '"223.999679_-68.727639".',
    )
    source_args.add_argument(
        "-f",
        "--sources_file",
        type=str,
        default=None,
        help="A file containing a list of sources to find. Each source should be "
        + "listed on a new line. The source format is the same as the -s option.",
    )

    # Observation arguments
    obs_args = parser.add_argument_group(
        "Observation arguments",
        "Options to specify the observation(s) to search. The default is all VCS "
        + "observations.",
    )
    obs_args.add_argument(
        "-o",
        "--obsids",
        type=int,
        nargs="*",
        default=None,
        help="A list of obs IDs to search.",
    )
    obs_args.add_argument(
        "--start",
        type=float,
        default=0.,
        help="Start time of the search, as a fraction of the full observation.",
    )
    obs_args.add_argument(
        "--end",
        type=float,
        default=1.,
        help="End time of the search, as a fraction of the full observation.",
    )

    # Functionality arguments
    finder_args = parser.add_argument_group(
        "Finder functionality arguments",
        "Options to specify how source(s) or observation(s) are found.",
    )
    finder_args.add_argument(
        "--obs_for_source",
        action="store_true",
        help="Find observations that each source is in.",
    )
    finder_args.add_argument(
        "--source_for_all_obs",
        action="store_true",
        help="Force a search for sources in all obs IDs.",
    )
    finder_args.add_argument(
        "--dt", type=float, default=60, help="Step size in time for beam modelling."
    )
    finder_args.add_argument(
        "--min_power",
        type=float,
        default=0.3,
        help="Minimum power to count as in the beam. If a normalisation mode is "
        + "selected, then this will be interpreted as a normalised power.",
    )
    finder_args.add_argument(
        "--norm_mode",
        type=str,
        choices=["zenith", "beam"],
        default="zenith",
        help="Beam power normalisation mode. 'zenith' will normalise to power at zenith. "
        + "'beam' will normalise to the peak of the primary beam.",
    )
    finder_args.add_argument(
        "--freq_mode",
        type=str,
        choices=["low", "centre", "high"],
        default="centre",
        help="Which frequency to use to compute the beam power. 'low' will use the "
        + "lowest frequency (most generous). 'centre' will use the centre frequency. "
        + "'high' will use the highest frequency (most conservative).",
    )

    args = parser.parse_args()

    # Initialise the logger
    logger = logger_setup.get_logger(loglevels[args.loglvl])

    if (
        not args.sources
        and not args.sources_file
        and not args.obsids
        and not args.source_for_all_obs
    ):
        logger.error("No sources or observations specified.")
        logger.info("If you would like to search for all sources in all obs IDs, "
                     + "use the --source_for_all_obs option.")
        sys.exit(1)

    if args.min_power < 0.0 or args.min_power > 1.0:
        logger.error("Normalised power must be between 0 and 1.")
        sys.exit(1)

    if args.norm_mode == "beam":
        logger.error("'beam' normalisation mode is not yet implemented.")
        sys.exit(1)

    if args.obsids is None and not args.obs_for_source and not args.source_for_all_obs:
        logger.error("No obs IDs specified while in source-for-obs mode.")
        logger.info("If you would like to search for sources in all obs IDs, "
                     + "use the --source_for_all_obs option.")
        sys.exit(1)

    if args.obs_for_source and (args.start != 0. or args.end != 1.):
        logger.error("Custom start and end time not available in obs-for-source mode.")
        sys.exit(1)

    # Decide whether to parse user provided source list or use the full catalogue
    if args.sources or args.sources_file:
        # Get sources from command line
        sources = args.sources
        if args.sources_file:
            logger.info("Parsing the provided source list file...")
            # Get sources from the provided source list file
            sources_from_file = np.loadtxt(args.sources_file, dtype=str, unpack=True)
            sources_from_file = list(sources_from_file)
            # If sources are found in the file, append them to the source list
            if sources:
                sources += sources_from_file
            else:
                sources = sources_from_file
    else:
        sources = None

    # Run the source finder
    (
        finder_result,
        beam_coverage,
        pointings,
        obs_metadata_dict,
        freq,
    ) = finder.find_sources_in_obs(
        sources,
        args.obsids,
        args.start,
        args.end,
        obs_for_source=args.obs_for_source,
        input_dt=args.dt,
        norm_mode=args.norm_mode,
        min_power=args.min_power,
        freq_mode=args.freq_mode,
        logger=logger,
    )

    if args.obs_for_source:
        file_output.write_output_source_files(
            finder_result,
            obs_metadata_dict,
            args.freq_mode,
            args.norm_mode,
            args.min_power,
            logger=logger,
        )

        file_output.plot_power_vs_time(
            pointings,
            obs_metadata_dict,
            beam_coverage,
            args.min_power,
            logger=logger,
        )
    else:
        file_output.write_output_obs_files(
            finder_result,
            obs_metadata_dict,
            args.start,
            args.end,
            freq,
            args.norm_mode,
            args.min_power,
            logger=logger,
        )
