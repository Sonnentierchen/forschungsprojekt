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

# Code Structure

The code is structure into different modules:

## Modifications to Mask RCNN implementation

The following python files were modified to account for needs in my project:

* visualize.py - added option to leave out masks
* coco.py - added year option to be able to load COCO datasets of different years

## Play Around

This code was written to get a first hold on the network implementation. This part consisted of the following classes:

* calc_bbs.py - to run the net on some image examples and visually assess the quality of the bounding boxes
* extract_frames.py - to extract images form the videos in combination with
* manual_frame_extraction.py - which called the function from the file above with specific paths

## Precision Assessment

This code was written to compare the power of the Mask RCNN implementation trained on the MS COCO dataset with the precision on manually
annotated frames from the VUFO videos.

The following classes play a role in this task:

* compare_coco_vufo_precision.py - to load the MS COCO dataset and use the coco evaluate function on it, as well as on the videos
* transform_vufo_to_coco_format.py - to transform a given video annotation file into a format that coco evaluate can process

Example call to compare_coco_vufo_precision.py:

```
python compare_coco_vufo_precision.py -v ./assets/input/videos/Video.3gp -videoOutput ./assets/output/vufo/ -y 2017 -coco ./assets/input/ --cocoOutput ./assets/output/ --limit 30
```

# Notes

Here is a collection of what to pay attention to when using the network.

* Input images have to in **.png** format, .jpgs are not accepted, but the error occurs later when the shape of the image is to be accessed.
* If Mask RCNN is cloned fresh and put in the mrcnn folder, there has to be a change made to modely.py: instead of `import utils` the line needs to be `from mrcnn.utils import utils`
* matplotlib cannot plot using SSH on the computing machine, since there's not XServer running. To fix this, add this at the top of the file using the matplotlib: `import matplotlib
matplotlib.use('Agg')`