# Bright Field Drift Correction

Use bright field signal to trace the drift in 3D with nanometer precision.

## Requirements
* python 3.5+
* numpy
* scipy
* scikit-image

## Installation
`git clone https://gitlab.pasteur.fr/aaristov/Brightfield_Drift_Tracking_3D.git`

`python setup.py install`

## Usage

In order to trace the drift in 3D a calibration stack needed (dict.tif).

Acquire bright field images with the same exposure (movie.tif) in order to trace the drift with nanometer precision.

### Tracing drift:
`python -m bfdc.Drift trace dict.tif movie.tif`

### Applying drift to ZOLA table:
`python -m bfdc.Drift trace ZOLA_localization_table.csv BFCC_table.csv`

### Help and possible arguments:

`python -m bfdc.Drift -h`

`python -m bfdc.Drift trace -h`

`python -m bfdc.Drift apply -h`
