# mwa_source_finder
Code to find sources in MWA VCS observations.

## Installation
To install, clone the repository and navigate to the source code directory, then run

    pip install .

Once installed, the command line interface can be accessed with `source-finder`.

## Examples
To search for all pulsars in an observation, run

    source-finder -o <obs ID>

To search for all VCS observations with a pulsar in it, run

    source-finder --obs_for_source -s <NAME>

The `-s` option also accepts pointings as `RA_DEC` coordinates.