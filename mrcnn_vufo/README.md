# Mask RCNN with VUFO classes

The code in `/mrcnn_vufo/` has been modified to work with VUFO classes only. The configuration files for the network indicate only 9 classes, i.e. background and the 8 VUFO classes. There are scripts in this folder to remove the weights that are related to the non-VUFO classes and would prevent loading the pre-trained weights. This is not an issue if the last layers of the bbox branch are omitted as in that case the wrongly shaped weights of those layers are not loaded at all. Another important change is the stop gradient node at the beginning of the mask head branch to prevent the weights from changing and adjusting to the non-existant masks in VIA data.

Since the inference, evaluation and training scripts work just as the ones in `mrcnn_coco` they are not explained here again. The key differences are the possibility to adjust a weights file and the changed number of classes.

## Adjust weights

If you don't want to leave out the last layers of a pre-trained weights file you can use the script in adjust_weights.py which will omit any information in the weights related to the non-VUFO classes of index 8 and higher. To see what the script does it is best to inspect a weights file with the HDFViewer and look for the layers that are being edited in the script.

### Command explanation

`mrcnn_vufo/adjust_weights.py` deletes the rows and columns in the weights matrices that are related to the non-VUFO classes. For details see the script.