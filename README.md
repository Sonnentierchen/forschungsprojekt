# Forschungsprojekt

This project aims at using the Mask RCNN implementation available at https://github.com/matterport/Mask_RCNN, strip it of its masking functioning and only keep the bounding box detection. The application domain is videos filmed inside driving cars but is not limited to it. The model pre-trained on the MS Coco dataset is used as the training for masking positvely influences the bounding box detection capability.

# Setup

The following is supposed to describe the setup process to be able to run the Mask RCNN network (https://github.com/matterport/Mask_RCNN).

## Setting up the environment

* Download and install Anaconda (https://conda.io/docs/user-guide/install/download.html)
* Create a virtual environment with Python 3.5 
```
conda create -n tensorflow python=3.5
```
* Activate the environment: source activate tensorflow
* Do the following with activated source
```
conda install tensorflow
```
```
conda install matplotlib
```
```
conda install scikit-image
```
```
conda install opencv
```
```
conda install keras
```
```
conda install ipython
```
```
conda install cython
```
* Clone with git: https://github.com/cocodataset/cocoapi.git.
* cd into the cocoapi/PythonAPI folder
* execute 
```
make
```
* Still in the folder execute
```
python setup.py install
```

## Setting up the Mask RCNN implementation

* Clone with git: https://github.com/matterport/Mask_RCNN **Mask RCNN's code has to be in the same folder as the python script**
* Download the weights for the network at https://github.com/matterport/Mask_RCNN/releases and put them in the source folder of Mask RCNN

# Notes

Here is a collection of what to pay attention to when using the network.

* Input images have to in **.png** format, .jpgs are not accepted, but the error occurs later when the shape of the image is to be accessed.
