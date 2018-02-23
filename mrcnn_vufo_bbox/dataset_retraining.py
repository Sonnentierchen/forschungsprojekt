from mrcnn import coco
from mrcnn import config
from pycocotools.coco import COCO
import numpy as np

import os

class Config(config.Config):
    """Configuration for evaluation on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # VUFO has 8 classes

    GPU_COUNT = 1

class Dataset(coco.CocoDataset):
    def load_coco(self, imagesPath, groundTruthAnnotationsPath, year=None, subset=None, class_ids=None,
                  class_map=None, return_coco=True):
        # Override method to provide own loading of data
        # Thus we can omit year and do not need to concatenate paths as
        # dataset_dir is the actual dataset file already

        if not os.path.exists(groundTruthAnnotationsPath):
            raise ValueError("Annotations file does not exist.")

        coco = COCO(groundTruthAnnotationsPath)

        # Load all classes or a subset?
        if not class_ids:
        # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
                # Remove duplicates
                image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
            "coco", image_id=i,
            path=os.path.join(imagesPath, coco.imgs[i]['file_name']),
            width=coco.imgs[i]["width"],
            height=coco.imgs[i]["height"],
            annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=False)))
        if return_coco:
            return coco

    def load_bboxes(self, image_id):
        """Load classes and bboxes for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        class_ids: a 1D array of class IDs of the instance masks.
        boxes: bbox array [num_instances, (y1, x1, y2, x2)].
        """

        # The reduced version of load_mask - we did this to be able to omit mask loading

        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        boxes = np.zeros([len(annotations), 4], dtype=np.int32)
        for index in range(len(annotations)):
            annotation = annotations[index]
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                class_ids.append(class_id)
            bbox = annotation['bbox']
            x, y, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            boxes[index] = np.array([y, x, y + height, x + width]).astype(np.int32)
        if class_ids:
            class_ids = np.array(class_ids, dtype=np.int32)
            return class_ids, boxes
