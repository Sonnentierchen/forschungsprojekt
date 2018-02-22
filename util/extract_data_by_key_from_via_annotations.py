import os
import json

def extract_data_by_key_from_via_annotations(annotations_path, key, output_path):
    assert os.path.exists(annotations_path), "Annotations file does not exist."
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    annotations_file_name = os.path.basename(annotations_path)
    annotations_file_name, extension = os.path.splitext(annotations_file_name)
    modified_annotations_file_name = annotations_file_name + "_" + key + extension

    with open(annotations_path, "r") as json_file, open(os.path.join(output_path, modified_annotations_file_name), "w") as json_output_file:
        json_data = json.load(json_file)
        output_json_data = {entry : json_data[entry] for entry in json_data if key in entry}
        json.dump(output_json_data, json_output_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Extracts all data that as a key that matches the specified string and stores the resulting dict at the specified location.")
    parser.add_argument("--annotations_path",
                        "-a",
                        required=True,
                        metavar="/path/to/instance_annotations.json",
                        help="The path to the instance annotations file.")
    parser.add_argument("--key",
                        "-k",
                        required=True,
                        help="The key that the keys in the annotations file have to match to be copied to the new annotations file.")
    parser.add_argument("--output_path",
                        "-o",
                        required=True,
                        help="The path where to store the result")
    args = parser.parse_args()
    extract_data_by_key_from_via_annotations(args.annotations_path, args.key, args.output_path)