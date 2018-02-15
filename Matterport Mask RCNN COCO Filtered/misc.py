COCO_CLASSES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

VUFO_CLASSES = ['person', 'car', 'motorcycle', 'bus', 'train', 'truck', 'bicycle']

def filter_result_for_classes(result, classes):
     import numpy as np

     filteredResult = {}
     filteredResult['rois'] = []
     filteredResult['class_ids'] = []
     filteredResult['scores'] = []
     filteredResult['masks'] = result['masks']
     for index in range(0, len(result['class_ids'])):
          class_id = result['class_ids'][index]
          if class_id in classes:
               # The name of the COCO class is also in the VUFO classes, i.e. add the bounding box
               filteredResult['rois'].append(np.array(result['rois'][index][0:4]))
               filteredResult['class_ids'].append(result['class_ids'][index])
               filteredResult['scores'].append(result['scores'][index])
     filteredResult['rois'] = np.array(filteredResult['rois'])
     filteredResult['class_ids'] = np.array(filteredResult['class_ids'])
     filteredResult['scores'] = np.array(filteredResult['scores'])
     return filteredResult