import os
import json

NUM_VUFO_CLASSES = 8

def remove_non_vufo_classes_from_coco_data(coco_annotations_path, output_path):
    assert os.path.exists(coco_annotations_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    coco_annotations_file_name, extension = os.path.splitext(coco_annotations_path)
    vufo_annotations_file_name = coco_annotations_file_name + "_vufo" + extension

    with open(coco_annotations_path, "r") as coco_json_file, open(vufo_annotations_file_name, "w") as vufo_json_file:
        coco_json_data = json.load(coco_json_file)
        vufo_json_data = {}
        vufo_json_data["info"] = coco_json_data["info"]
        vufo_json_data["licenses"] = coco_json_data["licenses"]
        vufo_json_data["images"] = coco_json_data["images"]
        vufo_json_data["categories"] = []
        vufo_json_data["annotations"] = []

        for category in coco_json_data["categories"]:
            category_id = int(category["id"])
            if not category_id > NUM_VUFO_CLASSES:
                vufo_json_data["categories"].append(category)

        for annotation in coco_json_data["annotations"]:
            annotation_category_id = int(annotation["category_id"])
            if not annotation_category_id > NUM_VUFO_CLASSES:
                vufo_json_data["annotations"].append(annotation)

        json.dump(vufo_json_data, vufo_json_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Removes all non-VUFO (i.e. above index 8) classes from a COCO instance annotations file.")
    parser.add_argument("--annotationsPath",
                        "-a",
                        required=True,
                        metavar="/path/to/instance_annotations.json",
                        help="The path to the instance annotations file.")
    parser.add_argument("--outputPath",
                        "-o",
                        required=True,
                        help="The path where to store the results. In this folder the annotations file with the extension"
                             "_vufo.json will be saved.")
    args = parser.parse_args()
    remove_non_vufo_classes_from_coco_data(args.annotationsPath, args.outputPath)