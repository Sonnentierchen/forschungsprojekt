# Mask RCNN with VUFO classes

The code in `/mrcnn_vufo/` has been modified to work with VUFO classes only. The configuration files for the network indicate only 9 classes, i.e. background and the 8 VUFO classes. There are scripts in this folder to remove the weights that are related to the non-VUFO classes and would prevent loading the pre-trained weights. This is not an issue if the last layers of the bbox branch are omitted as in that case the wrongly shaped weights of those layers are not loaded at all. Another important change is the stop gradient node at the beginning of the mask head branch to prevent the weights from changing and adjusting to the non-existant masks in VIA data.

**NOTE** Since the inference, evaluation and training scripts work just as the ones in `mrcnn_coco` they are not explained here again. The key differences are the possibility to adjust a weights file and the changed number of classes. The usage of the scripts remains the same.

## Adjust weights

If you don't want to leave out the last layers of a pre-trained weights file you can use the script in adjust_weights.py which will omit any information in the weights related to the non-VUFO classes of index 8 and higher. To see what the script does it is best to inspect a weights file with the HDFViewer and look for the layers that are being edited in the script.

### Command explanation

`mrcnn_vufo/adjust_weights.py` deletes the rows and columns in the weights matrices that are related to the non-VUFO classes. For details see the script.

## Experiments

The experiments undertaken with the model in `mrcnn_vufo` were more elaborate than the ones in `mrcnn_coco`. Additionally to restricting the classes to classes required by VUFO, it was tried out whether better results could be achieved leaving out the weights of the last layer, i.e. the bounding box head branch that produces the final bouding box classifications and bounding box deltas, or by furhter training it. It was also evaluated whether it makes more sense to train the whole network or restrict the trainable layers to be the ones of the bounding box head branch. The parameters are all provided under the headline of the respective experiment. All experiments were run using the weights modified with the `adjust_weights.py` script, i.e. `mask_rcnn_vufo_2017_nov.h5` in `assets/pre_trained_weights`.

The results of the training, i.e. the weights are stored in the folder that describe the training parameters. Next to each weights there is the video7 of the VUFO dataset with the bounding boxes drawn into it, 3 extracted frames from the video and the evaluation file that stores the evaluation results of the network on the VUFO 1500 dataset.

### VUFO 400

The experiments in the `vufo_400_incorrect_conversion` folder were conducted on the VUFO dataset consisting of 400 images. Unfortunately, I figured out later that the script `conversion.py` contained an error, which caused the width or height of a bounding box to be negative in some rather rare cases. The network did not complain during training and probably also took into account the negative widths and heights. That's why the experiments conducted on the VUFO 400 dataset are probably obsolete. The error was later corrected in the VUFO 1500 dataset.

_All experiment were run using the data only for training, as the dataset was small and it was thought to reduce the small set to much if a validation set was sliced off._

_All evaluation metrics are from running the network on the VUFO 1500 dataset. This choice was made to show the network performance on the final and more relevant VUFO dataset._

#### Original

Here the dataset consisted of only the original 400 VUFO images without any data augmentation.

##### Training all layers without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.01`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.049
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.135
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.066
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.050
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.034
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.064
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.066
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.058
 ```

##### Training all layers and omitting the weights of the bbox head branch

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `weights of bbox head branch`<br/>
`learning rate`: `0.01`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.036
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.068
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.068
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.075
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.049
 ```

#### Original and augmented

For these experiments the VUFO 400 dataset was expanded using the data augmentor. This resulted in nearly 1200 images, consisting of the original ones, the same images randomly cropped and again with added noise.

##### Training all layers without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.068
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.117
 ```

##### Training all layers by omitting the weights of the bbox head branch

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `weights of bbox head branch`<br/>
`learning rate`: `0.001`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.049
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.019
 ```

##### Training the bbox head branch only and without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.065
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.210
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.138
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.061
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.087
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.088
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.123
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.172
 ```

##### Training the bbox head branch only and omitting the weights of the bbox head branch

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `weights of bbox head branch`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.070
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.096
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.066
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.119
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.194
 ```

#### Original and augmented and the COCO val set for validation

For this experiment, the validation part of the COCO 2017 dataste was added to keep the network from storing overfitted weights.

##### Training the bbox head branch only and without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.112
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.488
 ```

### VUFO 1500

The following experiments, which are located in the folder `vufo_1500_correct_conversion`, were conducted using the VUFO 1500 dataset with 1500 images and the corrected implementation of the conversion function.
Since the dataset expanded to nearly 4500 images, 1500 original images, 1500 cropped images and 1500 noisy images, the set was split into a training and validation set. As an orientation the new split of the COCO 2017 dataset was approximated by 90% training data and 10% validation data, i.e. about 4050 images of the VUFO dataset for training, and about 450 for validation. The split was achieved using the `split_via.py` script in the `util` folder, which randomly separates a given VUFO formatted annotations file.

#### Original and augmented and the COCO val set for validation

For this experiment, the COCO validation set was added as validation to keep the network from storing overfitted weights.

##### Training all layers without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.563
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.231
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.332
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.304
 ```

#### Original and augmented and the COCO val set for training

In these experiments, the VUFO 1500 dataset was enriched with the COCO valiation set for training. This was an experimental decision based on the assumption, that althought the network should not be trained with the validation data, it might keep the network from overtting towards the VUFO data and at the same time improve the network performance because it is trained with more data it hadn't been trained on before.

##### Training all layers without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `all`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.577
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.140
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.250
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.301
 ```

##### Training the bbox head branch only and without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

Evaluation on VUFO 1500:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.104
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.128
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.129
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.236
 ```