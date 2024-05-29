import argparse
import sys

import mwa_source_finder as sf


def load_items_from_file(filename: str) -> list:
    """Load items from a file which contains one item per line.

    Parameters
    ----------
    filename : `str`
        The name of the file to read.

    Returns
    -------
    valid_items : `list`
        A list of items stored as strings.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    # Strip spaces and newlines
    items = [line.strip() for line in lines]
    # Remove empty strings from empty lines
    valid_items = [item for item in items if item]
    return valid_items


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Find sources in MWA VCS observations.",
        add_help=False,
    )
    loglevels = sf.utils.get_log_levels()

    # Program arguments
    optional = parser.add_argument_group("Program arguments")
    optional.add_argument("-h", "--help", action="help", help="Show this help information and exit.")
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
        "--sources_file",
        type=str,
        default=None,
        help="A file containing a list of sources to find. Each source should be "
        + "listed on a new line. The source format is the same as the -s option.",
    )

    # Observation arguments
    obs_args = parser.add_argument_group(
        "Observation arguments",
        "Options to specify the observation(s) to search. The default is all VCS " + "observations.",
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
        "--obsids_file",
        type=str,
        default=None,
        help="A file containing a list of obs IDs to search. Each obs ID should be " + "listed on a new line.",
    )
    obs_args.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time of the search, as a fraction of the full observation.",
    )
    obs_args.add_argument(
        "--end",
        type=float,
        default=1.0,
        help="End time of the search, as a fraction of the full observation.",
    )
    obs_args.add_argument(
        "--filter_available",
        action="store_true",
        help="Only search observations with data files available. "
        + "This will increase the time taken to obtain the metadata.",
    )
    obs_args.add_argument(
        "--no_cache",
        action="store_true",
        help="Do not read or write to the metadata cache.",
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
    finder_args.add_argument("--dt", type=float, default=60, help="Step size in time for beam modelling.")
    finder_args.add_argument(
        "--min_power",
        type=float,
        default=0.3,
        help="Minimum normalised power to count as in the beam.",
    )
    # finder_args.add_argument(
    #     "--norm_mode",
    #     type=str,
    #     choices=["zenith", "beam"],
    #     default="zenith",
    #     help="Beam power normalisation mode. 'zenith' will normalise to power at zenith. "
    #     + "'beam' will normalise to the peak of the primary beam [not implemented].",
    # )
    finder_args.add_argument(
        "--freq_mode",
        type=str,
        choices=["low", "centre", "high"],
        default="centre",
        help="Which frequency to use to compute the beam power. 'low' will use the "
        + "lowest frequency (most generous). 'centre' will use the centre frequency. "
        + "'high' will use the highest frequency (most conservative).",
    )

    # Output arguments
    out_args = parser.add_argument_group(
        "Output arguments",
        "Options to specify how the results are output. The minimum output is a text "
        + "file for each obs ID (in source-for-obs mode) or each source (in "
        + "obs-for-source mode).",
    )
    out_args.add_argument(
        "--beam_plot",
        action="store_true",
        help="Make a plot of the source path through the beam for each obs ID/source " + "combination.",
    )
    out_args.add_argument(
        "--time_plot",
        action="store_true",
        help="Make a plot of the source power over time for each obs ID for each "
        + "source. Only available in obs-for-source mode.",
    )
    out_args.add_argument(
        "--plan_obs_length",
        type=float,
        default=None,
        help="Find the best observation for a source based on the mean power, then "
        + "plan the start and stop times of an observation of the specified length, "
        + "in seconds. Only available in obs-for-source mode.",
    )
    out_args.add_argument(
        "--download_plan",
        action="store_true",
        help="When used in combination with --plan_obs_length, will find the best"
        + "observing times for each source, then find the 'contiguous' time intervals "
        + "during each observation when at least one source is in its optimal "
        + "observing period. An interval is considered 'non-contiguous' when no "
        + "sources are in their optimal observing period for more than 10 min.",
    )

    args = parser.parse_args()

    # Initialise the logger
    logger = sf.utils.get_logger(loglevels[args.loglvl])

    # Input checking
    if (
        not args.sources
        and not args.sources_file
        and not args.obsids
        and not args.obsids_file
        and not args.source_for_all_obs
    ):
        logger.error("No sources or observations specified.")
        logger.error(
            "If you would like to search for all sources in all obs IDs, " + "use the --source_for_all_obs option."
        )
        sys.exit(1)

    if args.min_power < 0.0 or args.min_power > 1.0:
        logger.error("Normalised power must be between 0 and 1.")
        sys.exit(1)

    norm_mode = "zenith"
    # if args.norm_mode == "beam":
    #     logger.error("'beam' normalisation mode is not yet implemented.")
    #     sys.exit(1)

    if args.obsids is None and args.obsids_file is None and not args.obs_for_source and not args.source_for_all_obs:
        logger.error("No obs IDs specified while in source-for-obs mode.")
        logger.error(
            "If you would like to search for sources in all obs IDs, " + "use the --source_for_all_obs option."
        )
        sys.exit(1)

    if args.obs_for_source and (args.start != 0.0 or args.end != 1.0):
        logger.error("Custom start and end time not available in obs-for-source mode.")
        sys.exit(1)

    if not args.obs_for_source and args.plan_obs_length:
        logger.warning("The --plan_obs_length option will do nothing in " + "source-for-obs mode.")

    if not args.obs_for_source and args.download_plan:
        logger.warning("The --download_plan option will do nothing in " + "source-for-obs mode.")

    # Get sources from command line, if specified
    if args.sources or args.sources_file:
        sources = args.sources
        if args.sources_file:
            logger.info(f"Parsing the provided source list file: {args.sources_file}")
            sources_from_file = load_items_from_file(args.sources_file)
            if sources:
                sources += sources_from_file
            else:
                sources = sources_from_file
    else:
        sources = None

    # Get obs IDs from command line, if specified
    if args.obsids or args.obsids_file:
        obsids = [str(obsid) for obsid in args.obsids]
        if args.obsids_file:
            logger.info(f"Parsing the provided obs ID list file: {args.obsids_file}")
            obsids_from_file = load_items_from_file(args.obsids_file)
            if obsids:
                obsids += obsids_from_file
            else:
                obsids = obsids_from_file
    else:
        obsids = None

    # Run the source finder
    (
        finder_results,
        beam_coverage,
        pointings,
        all_obs_metadata,
    ) = sf.find_sources_in_obs(
        sources,
        obsids,
        args.start,
        args.end,
        obs_for_source=args.obs_for_source,
        filter_available=args.filter_available,
        input_dt=args.dt,
        norm_mode=norm_mode,
        min_power=args.min_power,
        freq_mode=args.freq_mode,
        no_cache=args.no_cache,
        logger=logger,
    )

    if args.obs_for_source:
        if args.plan_obs_length is not None:
            obs_plan = sf.find_best_obs_times_for_sources(
                pointings.keys(),
                all_obs_metadata,
                beam_coverage,
                obs_length=args.plan_obs_length,
                logger=logger,
            )

            if args.download_plan:
                sf.plan_data_download(obs_plan, savename="download_plan.csv", logger=logger)
        else:
            obs_plan = None

        sf.write_output_source_files(
            finder_results,
            all_obs_metadata,
            args.freq_mode,
            norm_mode,
            args.min_power,
            obs_plan=obs_plan,
            logger=logger,
        )
    else:
        sf.write_output_obs_files(
            finder_results,
            all_obs_metadata,
            args.start,
            args.end,
            norm_mode,
            args.min_power,
            logger=logger,
        )

    if args.time_plot:
        sf.plot_power_vs_time(
            pointings.keys(),
            all_obs_metadata,
            beam_coverage,
            args.min_power,
            args.obs_for_source,
            logger=logger,
        )

    if args.beam_plot:
        # Ensure finder results are in source-for-obs format
        if args.obs_for_source:
            obs_finder_results = sf.invert_finder_results(finder_results)
        else:
            obs_finder_results = finder_results

        for obsid in all_obs_metadata:
            obs_metadata = all_obs_metadata[obsid]
            sf.plot_beam_sky_map(
                obs_finder_results[obsid],
                beam_coverage,
                obs_metadata,
                pointings,
                min_power=args.min_power,
                norm_to_zenith=True,
                logger=logger,
            )
