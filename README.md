# Forschungsprojekt

This project uses the Mask RCNN implementation available at https://github.com/matterport/Mask_RCNN. The reason why this network was chosen was because the [underlying network](https://arxiv.org/abs/1703.06870) was performing very well in the [MS COCO leaderboard](http://cocodataset.org/#detections-leaderboard). The authors stated that they observed a positive influence of the mask branch on the bounding box branch which contributed to the good performance. Unfortunately, the authors of the paper implemented their network in Caffe but the common sense in the group of the project was to use [Tensorflow](https://www.tensorflow.org/). That's why some research on an implementation in Tensorflow was done and Matterport's implementation was used as a basis. At the start of the project the performance of the network was not known, as the programmers did not provide any evaluation metrics. It could only be seen later, that the network was not performing as good as the implementation by the authors of the paper. It was then tried to improve the network's quality on the VUFO data with various experiments.

# Notes

Since there are a couple of things that need attention and a notes list at the end of the readme would maybe be overlooked they are stated here upfront.

* matplotlib cannot plot using SSH on the computing machine, since there's not XServer running. To fix this, add this at the top of the file using the matplotlib: `import matplotlib matplotlib.use('Agg')`. This is only important if you're adding new plotting functionality. This has already been implemented in the visualize.py script.
* Ignore that opencv's VideoCapture outputs `Wrong sample count`, extraction of videoframes works nonetheless.
* The error above seems to be due to some of the data of the videos being corrupt. If you run inference on the whole video, keep this in mind as the inference might crash seemingly randomly at a certain frame. If that is so, and it crashes at e.g. frame 1201 then set the limit to 1200. Refer to the readmes to see how to set the limit.
* If you try to run the network and there is an error in the PyCOCO tools that says that `unicode` is unkown, replace `unicode` with `str` and `str` with `byte` in the file that threw the error. This should solve that issue.
* If your training doesn't work check whether you used the converted annotations file and not the one in VIA format.
* If you're trying to perform inference on a whole video with a video file and receive the following error: `ValueError: not enough values to unpack (expected 2, got 0)` then the video was not found. Check that the file exists.

# The network

## Structure

To get a detailed impression of the network's structure please refer to the paper. A high level sketch is provided here:
```
           input + anchors
                |
            ResNet-101
         -----------------
         |               |
    ROI branch    Objectness classification 
         |             branch
         |               |
   Bounding Box     Classification branch
   Deltas branch
```
The mask branch is omitted here because it's not relevant to us. The network uses the input images together with a fixed set of pre-generated anchors that resemble bounding boxes of different size and location. The ROI branch computes deltas for the anchors and is trained minimizing the distance between the closest groundtruth bounding box and the anchor with the applied delta. The bounding box delta branch then refines this delta again and during training the loss between the optimal detla of the groundtruth bounding box and the region proposal of the ROI branch is minimized. On the other side the objectness classification branch evaluates whether an anchor holds an object. The actual classification whether an object is a car or something else takes place in the final classification branch.

## Weights

The initial pre-trained weights of the network were taken from https://github.com/matterport/Mask_RCNN/releases and resemble the state of November 2017.

# The structure of the project

The project is structure into several folders which we want to explain here. Each folder contains another README explaining the functions and everything else in detail. The structure of the training results is explained in the README of the respective network folder, e.g. in the README in `mrcnn_vufo`.

## Installation

This folder contains a README on how to install everything needed to make the network run as well as the dependencies file need by Anaconda.

## Assets

The assets folder contains the VUFO input data with the VIA annotations as well as the pre-trained weights. The pre-trained weights were downloaded from https://github.com/matterport/Mask_RCNN/releases and represent the state of November 2017. `mask_rcnn_coco_2017_nov.5` resembles the original file downloaded from the source, whereas `mask_rcnn_vufo_2017_nov.h5` are the weights with the modified weights matrix that does not contain any weights related to non-VUFO classes anymore. Non-VUFO classes are the classes with an index higher than 9. Only use this file with `mrcnn_vufo` or `mrcnn_vufo_no_mask_branch`.

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

_Since it does not make any sense to keep the smaller dataset it has been omitted and only the larger one is shipped with this project._

## VUFO 400

The dataset underwent several stages. The first stage consisted of 400 images manually annotated in VIA. After some training runs, I concluded that the bad quality of the results might also lie in the quality of the training data and I could verify that in COCO the bouding boxes were drawn a lot tighter than I had done. That's why I improved the quality of the bounding boxes again. But improving the quality was only possible to some extend, as the quality of the videos is rather poor and often cars could be recognized in the distant but only from the context of the whole image. E.g. if there is a black bump on the otherwise gray autobahn then that's a car. But a network might not be able to properly adapt to a 5 times 5 pixel car. Those cars have been annotated anyways since a human is capable of detecting those and that should be the goal of the network. The data augmentor was written to increase the number of new images.

## VUFO 1500

During training time I annotated more images and got nearly to 1500 images. This is the vufo_1500 dataset. By re-inspecting the annotated imags I could sometimes see small errors. This is why this dataset has to be treated with caution. I annotated the images with the greatest effort, but without a proper QA it is difficult to keep a steady quality. If you want to inspect my annotations you can use the VIA tool in the `via` folder.

To increase the size of the dataset, data augmentation was employed using the `data_augmentor.py` script in the `util` folder. For furhter details please refer to the README in the `util` folder. The augmentor crops images randomly or adds random noise. This way the dataset was expanded to nearly 4500 images and then, due to the increased amount of data, split into a training and validation set with a ration fo 90%. This was achieved using the `split_via.py` function which randomly splits a VIA-formatted annotations file. 

# The experiments

The goal of the experiments was to improve the network quality on the VUFO videos by adapting it to those specific videos. To reach this goal several different experiments with the non- and modified models were untertaken. 

The final results of experiments are stored within the `train` folder of the respective network folder, e.g. `mrcnn_vufo`. The details of the experiment are listed in the README of the network's folder. The naming of the folders indicates the state of the experiment that was applied.

All experiment folders have in common, that the final results (this means always the weights with the highest number of epochs in a folder) were again once run on a video, i.e. video7.3gp of the VUFO dataset and evaluated on the whole 1500 images VUFO set to give an example of the network's performance. The results are stored next to the weights of the experiment. The video.avi and instances.json are the artifacts of the inference run with the produced weights, and the evaluation.txt is the evaluation result on the 1500 images.

To inspect the training process simply open tensorboard with the parameter `--logdir=` set to the folder containing the weights and the events. It should then be possible to view the loss functions etc.