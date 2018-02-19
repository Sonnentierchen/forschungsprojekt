import os
import json

def remove_class_from_via_data(annotations_path, class_to_remove, output_path):
    assert os.path.exists(annotations_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    annotations_file_name, extension = os.path.splitext(annotations_path)
    removed_class_annotations_file_name = annotations_file_name + "_removed_region_attribute" + extension

    with open(annotations_path, "r") as json_file, open(removed_class_annotations_file_name, "w") as removed_class_json_file:
        json_data = json.load(json_file)
        for entry in json_data:
            entry_data = json_data[entry]
            for region in entry_data["regions"]:
                region_data = entry_data["regions"][region]
                region_attributes = region_data["region_attributes"]
                region_attributes.pop(class_to_remove, None)

        json.dump(json_data, removed_class_json_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Removes a specified region attribute from a given VIA format file because sometimes those attributes are added by " +
        "accident but can't be removed in the VIA editor.")
    parser.add_argument("--annotations_path",
                        "-a",
                        required=True,
                        help="The path to the annotations file where the class is to be removed from.")
    parser.add_argument("--class_to_remove",
                        "-c",
                        required=True,
                        help="The category class to remove.")
    parser.add_argument("--output_path",
                        "-o",
                        required=True,
                        help="The output path where to store the results.")
    args = parser.parse_args()
    remove_class_from_via_data(args.annotations_path, args.class_to_remove, args.output_path)