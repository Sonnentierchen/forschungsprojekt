# Utility functions

The collection of scripts in `/util/` emerged from different needs during the project. 

## Data augmentor

`data_augmentor.py` takes in VIA-formatted annotation files along with the folder holding the respectivate images and either adds noise to the images or croppes them randomly by cropping 20% at most of each side.

* `-i`: the path to the images
* `-a`: the path to the annotations file
* `-t`: the type of noise to be added - if not specified the images will be cropped. The noise types are "gauss", "s&p", "poisson" and "speckle"
* `-o`: the path to the folder where to store the results
* `--param1`: only used for noise, usage depends on the noise type
* `--param2`: only used for noise, usage depends on the noise type

## Extract data by key

`extract_data_by_key_from_via_annotations.py` extracts all annotations with the key matching the specified key. This can be useful, if a set of videos was annotated at the same time using the VIA tool but should now be separated.

* `-a`: the path to the annotations file
* `-k`: the key by which to separate the annotations
* `-o`: the path where to store the resulting annotations file

## Merge VIA data

`merge_via.py` can merge multiple VIA formatted annotations into one file. The usage was intended for use cases where to folders are all named alike, like "video1", "video2" etc.

## Split VIA data

`split_via.py` is there to split an existing set of VIA annotations into two sets of the specified split, e.g. 90 which corresponds to 90% and 10%. This can be helpful to split a VIA annotations file into a training and a validation set.

* `-a`: the path to the annotations file
* `-s`: the split to apply, e.g. 90
* `-o`: the output folder where to store the results

## Remove region attribute from VIA data

The script `remove_region_attribute_from_via_data.py` accounts for the lack of functionality of the VIA tool to remove accidentially added region attributes.

* `-a`: the path to the annotations file
* `-c`: the region attribute to remove
* `-o`: the output folder where to store the results

## Remove non-VUFO classes from COCO data

`remove_non_vufo_classes_from_coco_data.py` removes all COCO classes of index 9 and above, because only indices 0 - 8 correspond to the VUFO classes. This avoids problems with the code adjusted to the fewer classes.

## Extract frames

This script is able to extract frames from a video. You can specify how many frames should be extracted and whether they should be taken evenly from the video or simply from start to end until the limit is reached.

* `-vp`: the path to the video file
* `-op`: the output path
* `-n`: the maximum number of frames to be extracted - if set to 0 all frames are extracted
* `-d`: can be "even" or "none" and indicates whether frames are to be extracted evenly or from beginning to end