#!/usr/bin/env python

"""
Code to find sources in MWA VCS observations.

Author: C. P. Lee
Credits: N. Swainston and B. Meyers
"""

__version__ = '0.1'

import argparse

import numpy as np

from logger_module import get_log_levels, get_logger
from coord_module import get_pointings, get_atnf_pulsars


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        usage='%(prog)s [options]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Code to find sources in MWA VCS observations.',
        add_help=False
    )
    loglevels = get_log_levels()
    optional = parser.add_argument_group('Program arguments')
    optional.add_argument('-h', '--help', action='help',
        help='Show this help information and exit.')
    optional.add_argument('-V', '--version', action='version',
        version='%(prog)s {}'.format(__version__), help='Print version and exit.')
    optional.add_argument('-L', '--loglvl', type=str, choices=loglevels,
        default='INFO', help='Logger verbosity level.')
    source_args = parser.add_argument_group('Source arguments',
        'Options to specify the source(s) to find. The default is all pulsars.')
    source_args.add_argument('-s', '--sources', type=str, nargs='*', default = None,
        help='A list of sources to find. Sources can be specified as pulsar names ' + \
        'or equatorial coordinates separated by an underscore. Coordinates can be ' + \
        'in either decimal or sexigesimal format. For example, the following ' + \
        'arguments are all valid: "B1451-68" "J1456-6843" "14:55:59.92_-68:43:39.50" ' + \
        '"223.999679_-68.727639".')
    source_args.add_argument('-f', '--sources_file', type=str, default = None,
        help='A file containing a list of sources to find. Each source should be ' + \
        'listed on a new line. The source format is the same as the -s option.')
    obs_args = parser.add_argument_group('Observation arguments',
        'Options to specify the observation(s) to search. The default is all VCS observations.')
    obs_args.add_argument('-o', '--obsids', type=int, nargs='*', default = None,
        help='A list of obs IDs to search.')
    args = parser.parse_args()

    # Initialise the logger
    logger = get_logger(loglevels[args.loglvl])

    # Decide where to parse user provided source list or use the full catalogue
    if args.sources or args.sources_file:
        # Get sources from command line
        sources = args.sources
        if args.sources_file:
            logger.info('Parsing the provided source list...')
            # Get sources from the provided source list file
            try:
                sources_from_file = np.loadtxt(args.sources_file, dtype=str, unpack=True)
                sources_from_file = list(sources_from_file)
            except:
                logger.error(f'Cannot load file: {args.sources_file}')
            # If sources are found in the file, append them to the source list
            if sources:
                sources += sources_from_file
            else:
                sources = sources_from_file
        # Convert source list to pointing list
        pointings = get_pointings(sources, logger=logger)
        # Print out a full list of sources
        for pointing in pointings:
            logger.info(f"Source: {pointing['Name']:30} RAJ: {pointing['RAJ']:14} DECJ: {pointing['DECJ']:15}")
    else:
        logger.info('Collecting pulsars from the ATNF catalogue...')
        pointings = get_atnf_pulsars(logger=logger)
        logger.info(f'{len(pointings)} pulsars parsed from the catalogue')


if __name__ == "__main__":
    main()