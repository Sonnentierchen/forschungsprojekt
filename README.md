# Forschungsprojekt

This project aims at using the Mask RCNN implementation available at https://github.com/matterport/Mask_RCNN, strip it of its masking functioning and only keep the bounding box detection. The application domain is videos filmed inside driving cars but is not limited to it. The model pre-trained on the MS Coco dataset is used as the training for masking positvely influences the bounding box detection capability.

# Notes

Here is a collection of what to pay attention to when using the network.

* Input images have to in **.png** format, .jpgs are not accepted, but the error occurs later when the shape of the image is to be accessed.
* If Mask RCNN is cloned fresh and put in the mrcnn folder, there has to be a change made to modely.py: instead of `import utils` the line needs to be `from mrcnn.utils import utils`
* matplotlib cannot plot using SSH on the computing machine, since there's not XServer running. To fix this, add this at the top of the file using the matplotlib: `import matplotlib
matplotlib.use('Agg')`
* Ignore that opencv's VideoCapture outputs `Wrong sample count`, extraction of videoframes works nonetheless
* The videos are erroneous -> Set limit to frame that is corrupted
* PyCOCO tools error with str and unicodde -> replace str with byte and unicode with str

# The code structure

folder mrcnn is from the github repository
the rest is mine (most of the time)

# MS COCO

# The dataset

* non-VUFO classes are...

# The experiments

* explanation of structure of weights
*

tensorflow events -> tensorboard

# The util functions