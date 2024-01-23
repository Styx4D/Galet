# Galet
Official repository of GALET : A Deep Learning Image Segmentation Model for Drone-Based Grain Size Analysis of Gravel Bars

The repository is based on the implementation of Mask R CNN from Matterport https://github.com/matterport/Mask_RCNN

## Instalation
1 - Create a conda environment with python 3.6
2 - Install qgis by running
conda install -c conda-forge qgis
3 - Install requirements by running
pip install -r requirements.txt

Note : If you want to run the code on GPU, you will need to install tensorflow-gpu among with cudatoolkit and cudnn

## How to use
1 - Clone this repository
2 - Download the pretrained weight here : https://drive.google.com/file/d/18kRFTrrsK91y44fTgpr7q9e4fMKIAroQ/view?usp=sharing
3 - Put the weight on the folder GALET_RCNN_V3/logs/weight_GALET
4 - activate the environment and launch Qgis from it
conda activate *yout_env_name*
qgis
5 - Once in qgis, open the processing tool box and add custom script Qgis_processing_IMAGE.py and Qgis_processing_ORTHO.py. They will appear under Galet
[image 1]
6 - Open the python terminal and type
import keras
7 - Run GALET_georef for georeferenced orthomosaic, or GALET_image for ungeoreferenced image
[image 2]

### Example for georeferenced data
[video 1]





