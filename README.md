# Forschungsprojekt

This project aims at using the Mask RCNN implementation available at https://github.com/matterport/Mask_RCNN, strip it of its masking functioning and only keep the bounding box detection. The application domain is videos filmed inside driving cars but is not limited to it. The model pre-trained on the MS COCO dataset is used as the training for masking positvely influences the bounding box detection capability.

# Notes

Since there are a couple of things that need attention and a notes list at the end of the readme would maybe be overlooked they are stated here upfront.

* matplotlib cannot plot using SSH on the computing machine, since there's not XServer running. To fix this, add this at the top of the file using the matplotlib: `import matplotlib matplotlib.use('Agg')`. This is only important if you're adding new plotting functionality. This has already been implemented in the visualize.py script.
* Ignore that opencv's VideoCapture outputs `Wrong sample count`, extraction of videoframes works nonetheless.
* The error above seems to be due to some of the data of the videos being corrupt. If you run inference on the whole video, keep this in mind as the inference might crash seemingly randomly at a certain frame. If that is so, and it crashes at e.g. frame 1201 then set the limit to 1200. Refer to the readmes to see how to set the limit.
* If you try to run the network and there is an error in the PyCOCO tools that says that `unicode` is unkown, replace `unicode` with `str` and `str` with `byte` in the file that threw the error. This should solve that issue.

# The code structure

The project is structure into several folders which we want to explain here. Further details are in the readme files in the folders itself. The structure of the training results is explained further down under the experiments section.

## Installation

This folder contains a README on how to install everything needed to make the network run as well as the dependencies file need by Anaconda.

## Assets

The assets folder contains the VUFO input data with the VIA annotations as well as the pre-trained weights. The pre-trained weights were downloaded from https://github.com/matterport/Mask_RCNN/releases and represent the state of November 2017. Please pay attention to the extension `coco` or `vufo`, where `vufo` indicates that the weights of the original weights file that are related to the non-VUFO classes of COCO, i.e. classes with an index higher than 9 have been deleted. Only use this file with `mrcnn_vufo` or `mrcnn_vufo_no_mask_branch`.

## VIA

This folder contains a copy of the VIA annotation tool that can be run locally in a browser.

## MRCNN COCO

This folder contains Matterport's network in its nearly original state.

## MRCNN VUFO

This folder contains the network with the COCO classes removed.

## MRCNN VUFO No Mask Branch

This folder contains the heavliy modified network that has no mask branch anymore and does not use the mask loss for training.

# MS COCO

The project by Matterport uses the MS COCO dataset to train the network and COCO's evaluation tools. For furhter information visit their website at http://cocodataset.org.

# The dataset

The VUFO dataset consists of about 1500 hand-annoated images using the VIA tool. It should be noted that the images taken from the video are in many cases blurred and pixelated. Annotation was rather difficult and the quality of the annotations might have influenced the training results. To read further on this please refer to the report.

# The experiments

The final results of experiments are store within the `train` folder of the respective network. The details of the experiment are listed in the README of the network's folder. The naming of the folders indicates the state of the experiment that was applied.

All experiment folders have in common, that the final results were again once run on a video, i.e. video7.3gp of the VUFO dataset and evaluated on the whole 1500 images VUFO set. The results are stored next to the weights of the experiment. The video.avi and instances.json are the artifacts of the inference run with the produced weights, and the evaluation.txt is the evaluation result on the 1500 images.

To inspect the training process simply open tensorboard with the parameter `--logdir=` set to the folder containing the weights and the events. It should then be possible to view the loss functions etc.