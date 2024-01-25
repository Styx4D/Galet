# Galet
Official repository of [GALET : A Deep Learning Image Segmentation Model for Drone-Based Grain Size Analysis of Gravel Bars](https://www.researchgate.net/publication/362231914_GALET_A_deep_learning_image_segmentation_model_for_drone-based_grain_size_analysis_of_gravel_bars)

The repository is based on the implementation of  [Mask R CNN from Matterport](https://github.com/matterport/Mask_RCNN), modified to run on tensorflow 2.X

## Installation 

### With Conda
1 - Create a conda environment

2 - Install qgis by running
```bash
conda install -c conda-forge qgis
```
3 - Install requirements by running
```bash
pip install -r requirements.txt

```
Note : If you want to run the code on GPU, you will need to install cudatoolkit and cudnn
For example for tensorflow 2.1 :
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### With Osgeo
1 - Install Qgis with Osgeo (link). Use the advanced installation and select the package h5py

2 - Open the Osgeo Shell and install depedencies

Note : If you want to run the code on GPU, you will need to install cudatoolkit and cudnn from the official website.
You will also have to change the PATH of Qgis under options → environment to add :
NVIDIA GPU Computing Toolkit\CUDA\vxx.x\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxx.x\libnvvp

Please be aware of intalling compatible tensorflow/cuda/cudnn libraries

## How to use
1 - Clone this repository

2 - Download the pretrained weight [here](https://drive.google.com/file/d/18kRFTrrsK91y44fTgpr7q9e4fMKIAroQ/view?usp=sharing)

3 - Put the weight on the folder GALET_RCNN_V3/logs/weight_GALET

4 - If you make the intalation throw conda, activate the environment and launch Qgis from it. 
If you installed from Osgeow, just lunch Qgis.
```bash
conda activate *yout_env_name*
qgis
```

5 - Once in qgis, open the processing tool box and add script to toolbox `Qgis_processing_IMAGE.py` and `Qgis_processing_ORTHO.py`.
They will appear under Galet

![](img/Image2.png)


7 - Run GALET_georef for georeferenced orthomosaic, or GALET_image for ungeoreferenced image

![](img/Image1.png)

### Example for georeferenced data
TODO : video