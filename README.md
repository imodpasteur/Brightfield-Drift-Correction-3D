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

You'll need to select a contrast region say 32x32 picels from the dict.tif and save as crop.roi (ImageJ ROI format)

### Tracing drift:
`python -m bfdc.Drift trace dict.tif crop.roi movie.tif`

### Applying drift to ZOLA table:
`python -m bfdc.Drift trace ZOLA_localization_table.csv BFCC_table.csv --smooth 10`

### Help and possible arguments:

`python -m bfdc.Drift -h`

`python -m bfdc.Drift trace -h`

`python -m bfdc.Drift apply -h`

# Example of use
 
 Super resolution imaging of EDU Alexa 647 in the nucleus: beads photobleaching.
 
![input](img/sr_Substack%20(1-16384-1000)_l.gif) 
 
 Luckily, we have recorded the bright field movie along with the fluorescence by alternating illumination between illuimination and LED!
 
![input](img/bf_Substack%20(1-16384-1000)_l.gif) 
 
 How can we use it to trace the drift in 3D?
 
 Using the stack, recorded prior to imaging with the same exposure and same illumination intensity, we can reveal the drift from bright field only in 3D!

![input](img/dict_sr_crop.gif)

32x32 px crop form the dictionary and 64x64 crom form the movie tracks the drift with the precision higher than using beads and better sampling than redundant cross-corellation.

Stack crop ![input](img/dict_crop32.gif)
Movie crop ![input](img/bf_Substack%20(1-16384-1000)_crop32l.gif)


Note, how Nikon's Perfect Focus System struggles to maintain the focus!

![input](img/BFCC_table.csv_2zero.png) 

Fluorescent bead track before and after BFDC.

![input](img/bead_track_color.png) -> ![input](img/bead_track_color_BFDC.png) 
 

