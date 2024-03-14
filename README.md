# mwa_source_finder
Code to find sources in MWA VCS observations.

## Installation
To install, clone the repository and navigate to the source code directory, then run

    pip install --upgrade pip
    pip install .

`mwa_hyperbeam` requires the MWA FEE HDF5 file, which can be obtained with

    wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5

Then define the environment variable `MWA_BEAM_FILE` as the location of this file, e.g.

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