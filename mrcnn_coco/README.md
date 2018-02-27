# Mask RCNN with COCO classes

The code in `/mrcnn_coco/` is very close to the original GitHub repository with only small changes. The classes are kept the same in training, and only some rudimentary filtering to keep only VUFO classes in inference and evaluation are provided. This code was used to get to know the network and try out a few things.

## Inference

The following section explains how to perform inference with the code in `/mrcnn_coco/`, i.e. run the network on images.

### Command explanation

`mrcnn_coco/inference.py` runs the network on the specified image folder or video using the inidcated weights.

* `-w`: the path to the weights
* `-v`: the path to the video - if images folder is specified, inference is run on images and not on videos
* `-i`: the path to the images folder
* `-s`: indicates, whether also the images or video with the rendered bounding boxes should be saved alongside the COCO-formatted annotations file
* `-l`: the maximum number of images to process


### Command examples

`python mrcnn_coco/inference.py -w mrcnn_coco/train/coco/training_all_layers/0.001/weights_epoch_0100.h5 -v ./assets/input/vufo/original/videos/video7.3gp -o mrcnn_coco/train/coco/training_all_layers/0.001/unfiltered/ -s True -l 1200`

### Inference Filtered

`mrcnn_coco/inference_filtered.py` is the same function as stated above but filters out all non-VUFO classes.

## Evaluation

The following section explains how to perform evaluation with the code in `/mrcnn_coco/`.

### Command explanation

`mrcnn_coco/evaluation.py`  runs the network on the images at the given path, but only the ones that are included in the annotations file. It then computes the mean average precision (mAP) using the coco evaluation tools.

* `-w`: the path to the weights
* `-i`: the path to the images folder
* `-g`: the path to the ground truth file in COCO format
* `-o`: the output path where to store the evaluation text file
* `-m`: do not specify (legacy code)
* `-l`: the limit of how many images to run the evaluation on

### Command examples

The following command evaluates the weights that were trained on COCO data on the 1500 VUFO dataset.

`python mrcnn_coco/evaluation.py -w mrcnn_coco/train/coco/training_all_layers/0.001/weights_epoch_0100.h5 -i assets/input/vufo_1500/original/all_videos/ -g assets/input/vufo_1500/original/all_videos/all_videos_annotations_converted.json -o mrcnn_coco/train/coco/training_all_layers/0.001/unfiltered/ -l 1200`

### Evaluation Filtered

`mrcnn_coco/evaluation_filtered.py` is the same function as stated above but filters out all non-VUFO classes.

### Evaluation VIA and Evaluation VIA filtered

Both scripts are convenience methods so that the user does not have to convert the VIA-formatted annotation files first using the conversion.py script. This is done automatically and then the normal evaluation or filtered evaluation is run.

## Training

### Command explanation

### Command examples