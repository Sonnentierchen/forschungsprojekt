import os
import json
import cv2

# Unfortunately COCO has some left out indices etc. that's why we hard-code the categories here
COCO_CATEGORIES = [{"supercategory": "person", "id": 1, "name": "person"},
                   {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
                   {"supercategory": "vehicle", "id": 3, "name": "car"},
                   {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
                   {"supercategory": "vehicle", "id": 5, "name": "airplane"},
                   {"supercategory": "vehicle", "id": 6, "name": "bus"},
                   {"supercategory": "vehicle", "id": 7, "name": "train"},
                   {"supercategory": "vehicle", "id": 8, "name": "truck"},
                   {"supercategory": "vehicle", "id": 9, "name": "boat"},
                   {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
                   {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
                   {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
                   {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
                   {"supercategory": "outdoor", "id": 15, "name": "bench"},
                   {"supercategory": "animal", "id": 16, "name": "bird"},
                   {"supercategory": "animal", "id": 17, "name": "cat"},
                   {"supercategory": "animal", "id": 18, "name": "dog"},
                   {"supercategory": "animal", "id": 19, "name": "horse"},
                   {"supercategory": "animal", "id": 20, "name": "sheep"},
                   {"supercategory": "animal", "id": 21, "name": "cow"},
                   {"supercategory": "animal", "id": 22, "name": "elephant"},
                   {"supercategory": "animal", "id": 23, "name": "bear"},
                   {"supercategory": "animal", "id": 24, "name": "zebra"},
                   {"supercategory": "animal", "id": 25, "name": "giraffe"},
                   {"supercategory": "accessory", "id": 27, "name": "backpack"},
                   {"supercategory": "accessory", "id": 28, "name": "umbrella"},
                   {"supercategory": "accessory", "id": 31, "name": "handbag"},
                   {"supercategory": "accessory", "id": 32, "name": "tie"},
                   {"supercategory": "accessory", "id": 33, "name": "suitcase"},
                   {"supercategory": "sports", "id": 34, "name": "frisbee"},
                   {"supercategory": "sports", "id": 35, "name": "skis"},
                   {"supercategory": "sports", "id": 36, "name": "snowboard"},
                   {"supercategory": "sports", "id": 37, "name": "sports ball"},
                   {"supercategory": "sports", "id": 38, "name": "kite"},
                   {"supercategory": "sports", "id": 39, "name": "baseball bat"},
                   {"supercategory": "sports", "id": 40, "name": "baseball glove"},
                   {"supercategory": "sports", "id": 41, "name": "skateboard"},
                   {"supercategory": "sports", "id": 42, "name": "surfboard"},
                   {"supercategory": "sports", "id": 43, "name": "tennis racket"},
                   {"supercategory": "kitchen", "id": 44, "name": "bottle"},
                   {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
                   {"supercategory": "kitchen", "id": 47, "name": "cup"},
                   {"supercategory": "kitchen", "id": 48, "name": "fork"},
                   {"supercategory": "kitchen", "id": 49, "name": "knife"},
                   {"supercategory": "kitchen", "id": 50, "name": "spoon"},
                   {"supercategory": "kitchen", "id": 51, "name": "bowl"},
                   {"supercategory": "food", "id": 52, "name": "banana"},
	                 {"supercategory": "food", "id": 53, "name": "apple"},
                   {"supercategory": "food", "id": 54, "name": "sandwich"},
                   {"supercategory": "food", "id": 55, "name": "orange"},
                   {"supercategory": "food", "id": 56, "name": "broccoli"},
                   {"supercategory": "food", "id": 57, "name": "carrot"},
                   {"supercategory": "food", "id": 58, "name": "hot dog"},
                   {"supercategory": "food", "id": 59, "name": "pizza"},
                   {"supercategory": "food", "id": 60, "name": "donut"},
                   {"supercategory": "food", "id": 61, "name": "cake"},
                   {"supercategory": "furniture", "id": 62, "name": "chair"},
                   {"supercategory": "furniture", "id": 63, "name": "couch"},
                   {"supercategory": "furniture", "id": 64, "name": "potted plant" },
                   {"supercategory": "furniture", "id": 65, "name": "bed"},
                   {"supercategory": "furniture", "id": 67, "name": "dining table"},
                   {"supercategory": "furniture", "id": 70, "name": "toilet"},
                   {"supercategory": "electronic", "id": 72, "name": "tv"},
                   {"supercategory": "electronic", "id": 73, "name": "laptop"},
                   {"supercategory": "electronic", "id": 74, "name": "mouse"},
                   {"supercategory": "electronic", "id": 75, "name": "remote"},
                   {"supercategory": "electronic", "id": 76, "name": "keyboard"},
                   {"supercategory": "electronic", "id": 77, "name": "cell phone"},
                   {"supercategory": "appliance", "id": 78, "name": "microwave"},
                   {"supercategory": "appliance", "id": 79, "name": "oven"},
                   {"supercategory": "appliance", "id": 80, "name": "toaster"},
                   {"supercategory": "appliance", "id": 81, "name": "sink"},
                   {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
                   {"supercategory": "indoor", "id": 84, "name": "book"},
                   {"supercategory": "indoor", "id": 85, "name": "clock"},
                   {"supercategory": "indoor", "id": 86, "name": "vase"},
                   {"supercategory": "indoor", "id": 87, "name": "scissors"},
                   {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
                   {"supercategory": "indoor", "id": 89, "name": "hair drier"},
                   {"supercategory": "indoor", "id": 90, "name": "toothbrush"}
]

def mrcnn_instance_detections_to_coco_format(data):
    """
    Receives a dictionary returned by Matterport's Mask RCNN implementation
    and returns a dictionary in COCO format.
    data: the dictionary of the network

    INFO: The conversion method in coco.py by Matterport requires a dataset which we do not have
    here, thus this conversion method.
    """

    cocoData = []
    for entry in data:
        for index in range(0, len(entry['rois'])):
            roi = entry['rois'][index]
            cocoData.append({
                    "image_id" : entry["image_id"],
                    "category_id" : int(entry["class_ids"][index]),
                    # Bounding box is in [y1, x1, y2, x2] but COCO in [x, y, width, height]
                    "bbox" : [int(roi[1]), int(roi[0]), int(roi[3] - roi[1]), int(roi[2] - roi[0])],
                    "score" : round(float(entry["scores"][index]), 3)
                })

    return cocoData

def via_data_to_coco_evaluation_format(imagesPath, annotationsPath):
    """
    Loads the VUFO data file at the given location, extracts the information
    and stores it in a COCO compatible file format at the specified output
    location
    """

    cocoData = {}
    # Description for the data
    cocoData["info"] = {"description" : "Frames of videos from VUFO",
                        "url" : "",
                        "version" : "1.0",
                        "year" : "2018",
                        "contributor" : "",
                        "date_created" : ""}
    # Add a default license for all images
    cocoData["licenses"] = [{"url" : "",
                            "id" : "1",
                            "name" : "license"}]
    cocoData["images"] = []
    cocoData["annotations"] = []
    cocoData["categories"] = COCO_CATEGORIES

    imageIndex = 0
    annotationIndex = 0

    with open(annotationsPath) as annotationData:
        data = json.load(annotationData)
        for entry in data:
            entryData = data[entry]
            imageFile = entryData["filename"]
            image = cv2.imread(os.path.join(imagesPath, imageFile))
            width, height, channels = image.shape
            cocoData["images"].append({"license" : "1",
                               "file_name" : imageFile,
                               "coco_url" : "",
                               "height" : height,
                               "width" : width,
                               "date_captured" : "",
                               "flickr_url" : "",
                               "id" : imageIndex})

            regions = entryData["regions"]
            for regionIndex in regions:
                region = regions[regionIndex]
                regionAttributes = region["region_attributes"]
                coco_class_name = regionAttributes["class"]
                category = next((_category for _category in COCO_CATEGORIES if _category["name"] == coco_class_name), None)
                # Process region values
                shapeAttributes = region["shape_attributes"]
                x = int(shapeAttributes["x"])
                y = int(shapeAttributes["y"])
                # VIA data can be negative numbers
                x = x if x >= 0 else 0
                y = y if y >= 0 else 0
                width = int(shapeAttributes["width"])
                height = int(shapeAttributes["height"])
                # Hack, because the network expects masks we give it a mask in the form of the bounding box -> not problematic
                # since we stop the gradient with Kl.stop_gradient in the mask network head
                cocoData["annotations"].append({"segmentation" : [[x, y, x + width, y, x + width, y + width, x, y + width]],
                                                "area" : shapeAttributes["width"] * shapeAttributes["height"],
                                                "iscrowd" : 0,
                                                "image_id" : imageIndex,
                                                "bbox" : [x,
                                                          y,
                                                          width,
                                                          height],
                                                "category_id" : category["id"],
                                                "id" : annotationIndex})
                annotationIndex += 1

            imageIndex += 1
    
    return cocoData

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Perform inference on a set of images with pre-defined weights.")
    parser.add_argument("--imagesPath",
                        "-i", 
                        required=True, 
                        metavar="/path/to/images",
                        help="The path to the images.")
    parser.add_argument("--annotationsPath",
                        "-a",
                        required=True,
                        metavar="/path/to/annoations/",
                        help="The path to the annoations that are to be converted.")

    args = parser.parse_args()
    assert os.path.exists(args.imagesPath)
    assert os.path.exists(args.annotationsPath)

    result = via_data_to_coco_evaluation_format(args.imagesPath, args.annotationsPath)

    annotationsFileName = os.path.splitext(args.annotationsPath)[0]
    convertedPath = annotationsFileName + "_converted.json"
    with open(convertedPath, "w") as jsonFile:
        json.dump(result, jsonFile)