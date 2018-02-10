from mrcnn import coco
from mrcnn import config

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
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

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