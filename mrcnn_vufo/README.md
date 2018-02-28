# Mask RCNN with VUFO classes

The code in `/mrcnn_vufo/` has been modified to work with VUFO classes only. The configuration files for the network indicate only 9 classes, i.e. background and the 8 VUFO classes. There are scripts in this folder to remove the weights that are related to the non-VUFO classes and would prevent loading the pre-trained weights. This is not an issue if the last layers of the bbox branch are omitted as in that case the wrongly shaped weights of those layers are not loaded at all. Another important change is the stop gradient node at the beginning of the mask head branch to prevent the weights from changing and adjusting to the non-existant masks in VIA data.

**NOTE** Since the inference, evaluation and training scripts work just as the ones in `mrcnn_coco` they are not explained here again. The key differences are the possibility to adjust a weights file and the changed number of classes. The usage of the scripts remains the same.

## Adjust weights

If you don't want to leave out the last layers of a pre-trained weights file you can use the script in adjust_weights.py which will omit any information in the weights related to the non-VUFO classes of index 8 and higher. To see what the script does it is best to inspect a weights file with the HDFViewer and look for the layers that are being edited in the script.

### Command explanation

`mrcnn_vufo/adjust_weights.py` deletes the rows and columns in the weights matrices that are related to the non-VUFO classes. For details see the script.

## Experiments

### VUFO 400

The experiments in the `vufo_400_incorrect_conversion` folder were conducted on the VUFO dataset consisting of 400 images.

#### Original

##### Training all layers without omitting weights

##### Training all layers by omitting the weights of the bbox head branch

#### Original and augmented

##### Training all layers without omitting weights

##### Training all layers by omitting the weights of the bbox head branch

##### Training the bbox head branch only without omitting weights

##### Training the bbox head branch only omitting the weights of the bbox head branch

#### Original and augmented and the COCO val set for validation

##### Training the bbox head branch only without omitting weights

### VUFO 1500

#### Original and augmented and the COCO val set for validation

##### Training all layers without omitting weights

#### Original and augmented and the COCO val set for training

##### Training all layers without omitting weights

##### Training the bbox head branch only without omitting weights