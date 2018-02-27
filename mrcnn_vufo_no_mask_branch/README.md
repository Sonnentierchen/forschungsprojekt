# Mask RCNN with VUFO classes and the mask branch cut off

The purpose of this modification was to get rif off the mask branch and therefore prevent any influence on the rest of the weights despite the stop gradient node. This was achieved by omitting the loss function for the mask branch and also the head itself. The rest did not change and was kept similar to `mrcnn_vufo`. Results have proven to be stable that's why it makes the most sense to use this network instead of the others.

The commands are the same as in `mrcnn_coco` that's why they are not listed here again.