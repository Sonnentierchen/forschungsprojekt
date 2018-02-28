# Mask RCNN with VUFO classes and the mask branch cut off

The purpose of this modification was to get rif off the mask branch and therefore prevent any influence on the rest of the weights despite the stop gradient node. This was achieved by omitting the loss function for the mask branch and also the head itself. The rest did not change and was kept similar to `mrcnn_vufo`. Results have proven to be stable that's why it makes the most sense to use this network instead of the others.

**NOTE** Since the inference, evaluation and training scripts work just as the ones in `mrcnn_coco` they are not explained here again. The key differences are the same as the one in `mrcnn_vufo` but additionally and more importantly, the mask branch has been cut off.

## Experiments

### VUFO 1500

#### Original and augmented and the COCO val set for validation

##### Training the bbox head branch only without omitting weights

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `none`<br/>
`learning rate`: `0.0003`<br/>

##### Training the bbox head branch only and omitting the weights of the bbox head branch

Training parameters:<br/>
`runs`: `1`<br/>
`layers`: `bbox head branch only`<br/>
`omitted weights`: `weights of bbox head branch`<br/>
`learning rate`: `0.0003`<br/>