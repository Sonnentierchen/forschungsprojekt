import os
import json
import random
import copy

def split_via_annotations(annotations_path, split, output_path):
    assert os.path.exists(annotations_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(annotations_path, "r") as input_file, \
         open(os.path.join(output_path, "annotations_split" + str(split) + ".json"), "w") as first_file, \
         open(os.path.join(output_path, "annotations_split" + str(100 - split) + ".json"), "w") as second_file:
        input_data = json.load(input_file)
        key_list = copy.copy(list(input_data.keys()))
        first_count = (len(input_data) * split) / 100
        current_count = 0
        first_result = {}

        # Add random keys to the first dict until the percentage is satisfied
        while current_count < first_count:
            key = random.choice(key_list)
            first_result[key] = input_data[key]
            key_list.remove(key)
            current_count += 1

        # Add the rest to the second dict
        second_result = {entry : input_data[entry] for entry in key_list}
        json.dump(first_result, first_file)
        json.dump(second_result, second_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Split an exisiting VIA-formatted annotations file into the specified percentages.")
    parser.add_argument("--annotations_path",
        "-a",
        required=True,
        metavar="/path/to/the/annotations_file.json",
        help="The paths to the annotation file.")
    parser.add_argument("--split",
        "-s",
        required=True,
        help="The requested split in percent. E.g. 80 for a split into a file containing 80 percent of the data and one with 20.")
    parser.add_argument("--output_path",
        "-o",
        required=True,
        help="The path where the result will be stored as annotations.json")
    args = parser.parse_args()
    split = int(args.split)
    if split < 0: 
        split = 0
    if split > 100:
        split = 100
    split_via_annotations(args.annotations_path, split, args.output_path)