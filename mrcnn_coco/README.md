# Mask RCNN with COCO classes

The code in `/mrcnn_coco/` is very close to the original GitHub repository with only small changes. The classes are kept the same in training, and only some rudimentary filtering to keep only VUFO classes in inference and evaluation are provided. This code was used to get to know the network and try out a few things.

## Conversion

This script's purpose is to convert VIA-formatted data to the COCO format. The whole network as already adjusted to the COCO format and this way it is possible to use the COCO evaluation functions.

**IMPORTANT** The script computes masks in the shape of the bounding box to have any masks present and make the network run. This calls for further action to prevent the weights from changing towards these wrong masks. That's why at the beginning of the mask head branch a keras stop gradient node was inserted.

_Unfortunately, at the beginning of the experiments with the VUFO 400 dataset, the conversion function was faulty. See the experiments in the README of `mrcnn_vufo`. This error was corrected for the VUFO 1500 dataset._

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

```
python mrcnn_coco/inference.py -w mrcnn_coco/train/coco/training_all_layers/0.001/weights_epoch_0100.h5 -v ./assets/input/vufo/original/videos/video7.3gp -o mrcnn_coco/train/coco/training_all_layers/0.001/unfiltered/ -s True -l 1200
```

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

```
python mrcnn_coco/evaluation.py -w mrcnn_coco/train/coco/training_all_layers/0.001/weights_epoch_0100.h5 -i assets/input/vufo_1500/original/all_videos/ -g assets/input/vufo_1500/original/all_videos/all_videos_annotations_converted.json -o mrcnn_coco/train/coco/training_all_layers/0.001/unfiltered/ -l 1200
```

### Evaluation Filtered

`mrcnn_coco/evaluation_filtered.py` is the same function as stated above but filters out all non-VUFO classes.

## Training

The following section explains how to further train the network using the function `mrcnn_coco/training.py`.

### Command explanation

The command takes several parameters, but that makes it rather flexible.

* `-w`: the path to the weights
* `-l`: the path to the logs folder, i.e. where the resulting weights and events are stored at - you can open this folder with tensorboard and inspect the loss, etc.
* `--trainImagesPaths`: the path to the folder with the training images. You can specify multiple paths, but you also have to specify the same number of annotation paths. For the first path, the first annotations file is used, for the second, the second and so one. So be sure to properly match the images paths and annotation paths.
* `--trainAnnotationsPaths`: the annotation paths, each belonging to an images path. Be careful: The annotation files have to be in COCO format, not in the VIA format.
* `--valImagesPaths`: the path to the images used for validation. The same as for the training images holds for the validation images. You need to specify enough validation annotations and they are matched just like the training annotations.
* `--valAnnotationsPaths`: the annotation paths, each belonging to an images path. Be careful: The annotation files have to be in COCO format, not in the VIA format.
* `-r`: the number of runs. If you specify more than one run, you have to specify enough epochs (e.g. 50 100 200 if you set r to 3- no need to separate with comma), learning rates (e.g. 0.01 0.001 0.0001) and enough layers (e.g. heads 3+ all).
* `--epochs`: the number of epochs per run. The first number will be used for the first run, the second for the second, etc.
* `--learningRates`: the learning rate per run. The first number will be used for the first run, the second for the second, etc.
* `--layers`: the layers to train in each run. This can be a predefined key like "all" or "heads". To see what predefined keys are possible look into `mrcnn/model.py`. Other than that this key takes a regex per run. E.g. for `r=1` you could specify `"(mrcnn_class_logits|mrcnn_class|mrcnn_bbox_fc|mrcnn_bbox)"` to only train the network heads of the bounding box network head but not the mask network head. If you have multiple runs you have to specify multiple strings.

### Command examples

```
python mrcnn_coco/training.py -w assets/pre_trained_weights/mask_rcnn_coco_2017_nov.h5 -l assets/output/training/coco/2017/ --trainImagesPaths assets/input/coco/2017/train2017/ --trainAnnotationsPaths assets/input/coco/2017/annotations/instances_train2017.json --valImagesPaths assets/input/coco/2017/val2017/ --valAnnotationsPaths assets/input/coco/2017/annotations/instances_val2017.json -r 1 --epochs 100 --learningRates 0.0003 --layers all
```

## Experiments

Further training was conducted on the MS COCO 2017 dataset. This experiment was performed more of the purpose to get to know the network and a feeling for training than major results. As expected the loss was more jumpy than decreasing.

The training schedule was as suggested by the Matterport programmers:

`runs`: `3`
`layers`: `heads`, `4+`, `all`
`omitted weights`: `none`
`learning rate`: `0.001`, `0.001`, `0.0001`

The resulted weights are stored in the folder `train`.