# mwa_source_finder
Code to find sources in MWA VCS observations.

## Installation
This package uses [mwa_hyperbeam](https://github.com/MWATelescope/mwa_hyperbeam),
which requires the MWA FEE HDF5 file. This file can be obtained as follows:

    wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5

Then define the environment variable `MWA_BEAM_FILE` as the location of the HDF5 file:

    export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5

`mwa-source-finder` can be installed into your working environment using `pip`:

    pip install git+https://github.com/cplee1/mwa_source_finder.git

For most usage, the included command line interface (CLI) should be sufficient.
However, the package can be imported into Python as `mwa_source_finder`. The
CLI implementation is a good place to start to see how the package is used.

## CLI Usage
Once installed, the CLI can be accessed with `source-finder`. Refer to the help
menu (`-h`) for details on its usage and the option defaults.

The basic usage is as follows. The options in square brackets are the plot types
available. To search for all pulsars in an observation, run

    source-finder -o <obs ID> [--time_plot] [--beam_plot]

To search for all observations with a given source in it, run

    source-finder -s <source> --obs_for_source [--time_plot] [--beam_plot]

where the `-s` option accepts either pulsar names or `RA_DEC` coordinates.
The `-o` and `-s` options can also be used together to search for one or more
sources within one or more observations.

### Tips
By default, using `--obs_for_source` without any specified obs IDs will search
for the source(s) in all obs IDs in the archive (including deleted observations).
The `--filter_available` option can be used to check whether each observation has
data available in the archive. This additional step will increase the execution
time as it requires additional metadata checks.

Querying the metadata server for a large number of obs IDs can take a long time,
especially when searching the whole archive. Because of this, `mwa-source-finder` will
automatically cache obs ID metadata in a file called `sf_obsid_cache.yaml` in the
directory where the CLI is run. The metadata will be automatically read from the
cache the next time that the obs ID is searched. To disable reading/writing
to the cache file, use the `--no_cache` option.