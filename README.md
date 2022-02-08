[![Anaconda-Server Badge](https://anaconda.org/scipp/ess/badges/installer/conda.svg)](https://conda.anaconda.org/scipp)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Builds](https://github.com/scipp/ess/actions/workflows/weekly_and_release.yml/badge.svg)](https://github.com/scipp/ess/actions/workflows/weekly_and_release.yml)

# ess

European Spallation Source facility bespoke, neutron scattering tools based on
[scipp](https://github.com/scipp/scipp).
This is a collection of python tools providing facility and instrument specific support
for the ESS.

This supersedes scripts in [ess-legacy](https://github.com/scipp/ess-legacy)

This is built on and complemented by neutron and technique specific support available in
[scippneutron](https://github.com/scipp/scippneutron).

Please read the [contribution](contributing.md) guidelines before making additions.

# Branch organisation

Developments `release` are pinned to the latest
[stable release](https://github.com/scipp/scipp/tags) of scipp.
Other ongoing feature developments should be merged into `main`.
