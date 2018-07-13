# Bright Field Drift Correction

Use bright field signal to trace the drift in 3D with nanometer precision.

## Requirements
* python 3.6+
* numpy
* scipy
* scikit-image

## Installation

`pip install git+https://github.com/imodpasteur/Brightfield_Drift_Tracking_3D.git`

## Usage

In order to trace the drift in 3D a calibration stack needed (dict.tif).

Acquire bright field images with the same exposure (movie.tif) in order to trace the drift with nanometer precision.

You'll need to select a contrast region say 32x32 picels from the dict.tif and save as crop.roi (ImageJ ROI format)

### Tracing drift:
`python -m bfdc trace dict.tif crop.roi movie.tif`

### Applying drift to ZOLA table:
`python -m bfdc trace ZOLA_localization_table.csv BFCC_table.csv --smooth 10`

### Help and possible arguments:

`python -m bfdc -h`

`python -m bfdc trace -h`

`python -m bfdc apply -h`

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

# Change log

v0.1.2 Removing localizations from the ZOLA table if they come from bright field. 

v0.1.1 Automatically detecting BF frames based on intensity. No more need of skipping parameters to know.


 

