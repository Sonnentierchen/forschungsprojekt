# Mask RCNN with VUFO classes

The code in `/mrcnn_vufo/` has been modified to work with VUFO classes only. The configuration files for the network indicate only 9 classes, i.e. background and the 8 VUFO classes. There are scripts in this folder to remove the weights that are related to the non-VUFO classes and would prevent loading the pre-trained weights. This is not an issue if the last layers of the bbox branch are omitted as in that case the wrongly shaped weights of those layers are not loaded at all. Another important change is the stop gradient node at the beginning of the mask head branch to prevent the weights from changing and adjusting to the non-existant masks in VIA data.

**NOTE** Since the inference, evaluation and training scripts work just as the ones in `mrcnn_coco` they are not explained here again. The key differences are the possibility to adjust a weights file and the changed number of classes. The usage of the scripts remains the same.

## Adjust weights

If you don't want to leave out the last layers of a pre-trained weights file you can use the script in adjust_weights.py which will omit any information in the weights related to the non-VUFO classes of index 8 and higher. To see what the script does it is best to inspect a weights file with the HDFViewer and look for the layers that are being edited in the script.

### Command explanation

`mrcnn_vufo/adjust_weights.py` deletes the rows and columns in the weights matrices that are related to the non-VUFO classes. For details see the script.

## Experiments

The experiments undertaken with the model in `mrcnn_vufo` were more elaborate than the ones in `mrcnn_coco`. Additionally to restricting the classes to classes required by VUFO, it was tried out whether better results could be achieved leaving out the weights of the last layer, i.e. the bounding box head branch that produces the final bouding box classifications and bounding box deltas, or by furhter training it. It was also evaluated whether it makes more sense to train the whole network or restrict the trainable layers to be the ones of the bounding box head branch. The parameters are all provided under the headline of the respective experiment. All experiments were run using the weights modified with the `adjust_weights.py` script, i.e. `mask_rcnn_vufo_2017_nov.h5` in `assets/pre_trained_weights`.

### VUFO 400

The experiments in the `vufo_400_incorrect_conversion` folder were conducted on the VUFO dataset consisting of 400 images. Unfortunately, I figured out later that the script `conversion.py` contained an error, which caused the width or height of a bounding box to be negative in some rather rare cases. The network did not complain during training and probably also took into account the negative widths and heights. That's why the experiments conducted on the VUFO 400 dataset are probably obsolete. The error was later corrected in the VUFO 1500 dataset.

_All experiment were run using the data only for training, as the dataset was small and it was thought to reduce the small set to much if a validation set was sliced off._

#### Original

Here the dataset consisted of only the original 400 VUFO images without any data augmentation.

##### Training all layers without omitting weights

Training parameters:
`runs`: `1`
`layers`: `all`
`omitted weights`: `none`
`learning rate`: `0.01`

##### Training all layers by omitting the weights of the bbox head branch

Training parameters:
`runs`: `1`
`layers`: `all`
`omitted weights`: `weights of bbox head branch`
`learning rate`: `0.01`

#### Original and augmented

For these experiments the VUFO 400 dataset was expanded using the data augmentor. This resulted in nearly 1200 images, consisting of the original ones, the same images randomly cropped and again with added noise.

##### Training all layers without omitting weights

Training parameters:
`runs`: `1`
`layers`: `all`
`omitted weights`: `none`
`learning rate`: `0.0003`

##### Training all layers by omitting the weights of the bbox head branch

Training parameters:
`runs`: `1`
`layers`: `all`
`omitted weights`: `weights of bbox head branch`
`learning rate`: `0.001`

##### Training the bbox head branch only without omitting weights

Training parameters:
`runs`: `1`
`layers`: `bbox head branch only`
`omitted weights`: `none`
`learning rate`: `0.0003`

##### Training the bbox head branch only omitting the weights of the bbox head branch

Training parameters:
`runs`: `1`
`layers`: `bbox head branch only`
`omitted weights`: `weights of bbox head branch`
`learning rate`: `0.0003`

#### Original and augmented and the COCO val set for validation

For this experiment, the validation part of the COCO 2017 dataste was added to keep the network from storing overfitted weights.

##### Training the bbox head branch only without omitting weights

Training parameters:
`runs`: `1`
`layers`: `bbox head branch only`
`omitted weights`: `none`
`learning rate`: `0.0003`

### VUFO 1500

The following experiments were conducted using the VUFO 1500 dataset with 1500 images and the corrected implementation of the conversion function.
Since the dataset expanded to nearly 4500 images, 1500 original images, 1500 cropped images and 1500 noisy images, the set was split into a training and validation set. As an orientation the new split of the COCO 2017 dataset was approximated by 90% training data and 10% validation data, i.e. about 4050 images of the VUFO dataset for training, and about 450 for validation. The split was achieved using the `split_via.py` script in the `util` folder, which randomly separates a given VUFO formatted annotations file.

#### Original and augmented and the COCO val set for validation



##### Training all layers without omitting weights

#### Original and augmented and the COCO val set for training

In these experiments, the VUFO 1500 dataset was enriched with the COCO valiation set for training. 

##### Training all layers without omitting weights

##### Training the bbox head branch only without omitting weights