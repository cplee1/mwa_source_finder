# mwa_source_finder
Code to find sources in MWA VCS observations.

## Installation
This package can be installed into your working environment using `pip` with
the follwing command:

    pip install git+https://github.com/cplee1/mwa_source_finder.git

Alternatively, you can install the code from source by cloning the repository and
running

    pip install <source_directory>

Where `<source_directory>` is the directory containing the `pyproject.toml` file.

This code uses `mwa_hyperbeam`, which requires the MWA FEE HDF5 file. This file
can be obtained as follows:

    wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5

Then export the environment variable `MWA_BEAM_FILE` as the location of this file:

    export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5

Once installed, the command line interface can be accessed with `source-finder`.

## Examples
Here we show simple examples of how to use the command line interface. The options
in square brackets are the available plot types. Note that additional features are
detailed in the help information (`-h`).

To search for all pulsars in an observation, run

    source-finder -o <obs ID> [--beam_plot] [--time_plot]

To search for all observations with a given source in it, run

    source-finder --obs_for_source -s <NAME> [--beam_plot] [--time_plot]

where the `-s` option accepts pointings as either pulsar names or `RA_DEC` coordinates.

The `--filter_available` option can be used in obs-for-source mode to only search
observations which have data available in the MWA archive. This additional step will
increase the execution time as it requires additional metadata checks. By default,
the code will generate a cache file on first execution to save the observation
metadata for later use. This will save time on subsequent executions in obs-for-source
mode. The cache can be disabled with `--no_cache`.