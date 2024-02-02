# Galet
Official repository of [GALET : A Deep Learning Image Segmentation Model for Drone-Based Grain Size Analysis of Gravel Bars](https://www.researchgate.net/publication/362231914_GALET_A_deep_learning_image_segmentation_model_for_drone-based_grain_size_analysis_of_gravel_bars)

The repository is based on the implementation of  [Mask R CNN from Matterport](https://github.com/matterport/Mask_RCNN), modified to run on tensorflow 2.X.
Our approach is a local Server-Client Implementation. The user starts a Server (Mask-R-CNN) that will handle all of the image processing, while a QGis acts as a Client, and is used as the GUI and for the exploitation of segmentation results. This approach was driven by the difficulty of having QGis and TensorFlow work together properly into an unique Conda space. Please note that this is a localhost Client/Server approach. None of your data is sent on Internet.


## Preparing the installation

1 - Clone this repository

2 - Download the pretrained weight [here](https://drive.google.com/file/d/18kRFTrrsK91y44fTgpr7q9e4fMKIAroQ/view?usp=sharing) and put them into the folder GALET_RCNN_V3/weights/


## Mask-R-CNN Server installation

This installation procedure is based upon Conda.

1 - From an Anaconda Prompt, Create a python 3.8 or 3.9 conda environment and activate it
```bash
conda create -n galet_server python=3.9
conda activate galet_server
```
**You need every following step to be performed under the newly created galet_server conda environment**

2 - (skip this section if you don't want to, or can't run GALET on GPU)
Ensure that you have CUDA properly installed and setup by running
```bash
nvidia-smi
```
(if this command is not recognized, run the CUDA installation again and/or set the CUDA_PATH variable in your environment. Use Google for further instructions. If you change the PATH, you need to launch the Anaconda Prompt again (don't forget to activate the galet server environment) so that it is taken into account.
Then link your conda environment to CUDA
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

3 - From the root folder of this project, install the server libraries
```bash
pip install -r requirements-server.txt
```

4 - From the root folder of the project, launch the server to verify that everything works as expected
```bash
python Galet_Server.py
```
Leave this window open as long as you need to use the Mask-R-CNN network. You can shut down the server by hitting Ctrl+C. Don't forget that every time you want to start the server, you need to activate the galet server conda environment.


## QGis Client Installation

### With Conda
1 - From an Anaconda Prompt, create a conda environment (any Python version>3.8 should be suitable)
```bash
conda create -n galet_qgis
conda activate galet_qgis
```
**You need every following step to be performed under the newly created galet_qgis conda environment**

2 - Install qgis and the required client libraries by running
```bash
conda install -c conda-forge qgis Pillow rasterio shapely opencv rtree
```

3 - Launch QGis
```bash
qgis
```


## How to use
1 - Within the server environment, launch the Mask-R-CNN server

2 - Within the client environment, launch QGis

3 - Once in qgis, open the processing tool box and add script to toolbox `Qgis_processing_IMAGE.py` and `Qgis_processing_ORTHO.py`.
They will appear under Galet

![](img/Image2.png)


4 - Run GALET_georef for georeferenced orthomosaic, or GALET_image for ungeoreferenced image

![](img/Image1.png)

### Example for georeferenced data
TODO : video