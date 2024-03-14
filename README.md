# mwa_source_finder
Code to find sources in MWA VCS observations.

## Installation
To install, clone the repository and navigate to the source code directory, then run

    pip install .

Once installed, the command line interface can be accessed with `source-finder`.

## Examples
Here we show simple examples of how to use the command line interface. The opions
in square brackets are the available visualisations for the particular command.
Note that additional features are detailed in the help information (`-h`).

To search for all pulsars in an observation, run

    source-finder -o <obs ID> [--beam_plot] [--time_plot]

To search for all VCS observations with a source in it, run

    source-finder --obs_for_source -s <NAME> [--beam_plot] [--time_plot]

where the `-s` option accepts pointings as either pulsar names or `RA_DEC` coordinates.