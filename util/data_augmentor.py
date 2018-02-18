import cv2
import os
import shutil
import numpy as np
import json
import util
import random

def add_noise_to_images(images_path, annotations_path, output_path, noise_type, param_1, param_2):
    """
    Adds noise of the given type to all the images at the specified location and stores them at the specified output
    path together with a copy of the annotations file.

    :param images_path: the path to the images
    :param annotations_path: the path to the annotations of the images in VIA format
    :param output_path: the path where the results are to be stored
    :param noise_type: the type of noise to apply
    :param param_1: the first parameter of the noise, usage depends on the type of noise
    :param param_2: the second parameter of the noise, usage depends on the type of noise
    """
    assert os.path.exists(images_path)
    assert os.path.exists(annotations_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_file_names = util.get_images_at_path(images_path)

    # Filename of annotations file without path, e.g. annotations.json
    annotations_file_name, annotations_file_extension = os.path.splitext(os.path.basename(annotations_path))
    # Full path to new annotations file, e.g. /home/user/annotations/annotations_noisy.json
    modified_annotations_file_path = os.path.join(output_path, annotations_file_name + "_noisy" + annotations_file_extension)

    with open(annotations_path, "r") as json_file, open(modified_annotations_file_path, "w") as modified_json_file:
        json_data = json.load(json_file)

        for image_file_name in image_file_names:
            # Load the image to augment
            image = cv2.imread(os.path.join(images_path, image_file_name))

            noisy = None

            if noise_type == "gauss":
                row,col,ch= image.shape
                mean = float(param_1) # e.g. 0
                var = float(param_2) # e.g. 0.1
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)
                noisy = image + gauss
            elif noise_type == "s&p":
                row,col,ch = image.shape
                s_vs_p = float(param_1) # e.g. 0.5
                amount = float(param_2) # e.g. 0.004
                out = np.copy(image)
                # Salt mode
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
                out[coords] = 1

                # Pepper mode
                num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
                out[coords] = 0
                noisy = out
            elif noise_type == "poisson":
                vals = len(np.unique(image))
                vals = 2 ** np.ceil(np.log2(vals))
                noisy = np.random.poisson(image * vals) / float(vals)
            elif noise_type =="speckle":
                row,col,ch = image.shape
                gauss = np.random.randn(row,col,ch)
                gauss = gauss.reshape(row,col,ch)
                noisy = image + image * gauss
            else:
                raise ValueError("Unkown noise type.")

            if not noisy is None:
                print("Adding noise to image {}.".format(image_file_name))
                image_file_base_name, extension = os.path.splitext(image_file_name)
                augmented_file_name = image_file_base_name + "_noisy" + extension
                augmented_file_path = os.path.join(output_path, augmented_file_name)
                cv2.imwrite(augmented_file_path, noisy)
                # VIA format has the file size in the dictionary
                statinfo = os.stat(augmented_file_path)
                for key in json_data:
                    if image_file_name in key:
                        new_key = augmented_file_name + str(statinfo.st_size)
                        image_data = json_data[key]
                        if not image_data is None:
                            # In case that we read an image that is not in the dict we skip it
                            image_data["filename"] = augmented_file_name
                            image_data["size"] = int(statinfo.st_size)
                            json_data[new_key] = json_data.pop(key)

        json.dump(json_data, modified_json_file)

def crop_and_resize_images(images_path, annotations_path, output_path):
    """
    Augments the images at the specified path by cropping and resizing them and stores the results together with the
    annotations file at the specified location.

    :param images_path: the path to the images
    :param annotations_path: the path to the annotations file in VIA format
    :param output_path: the path where the results are to be stored
    """
    assert os.path.exists(images_path)
    assert os.path.exists(annotations_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_file_names = util.get_images_at_path(images_path)

    # Filename of annotations file without path, e.g. annotations.json
    annotations_file_name, annotations_file_extension = os.path.splitext(os.path.basename(annotations_path))
    # Full path to new annotations file, e.g. /home/user/annotations/annotations_noisy.json
    modified_annotations_file_path = os.path.join(output_path, annotations_file_name + "_cropped" + annotations_file_extension)

    with open(annotations_path, "r") as json_file, open(modified_annotations_file_path, "w") as modified_json_file:
        json_data = json.load(json_file)

        for image_file_name in image_file_names:

            # Load the image to augment
            image = cv2.imread(os.path.join(images_path, image_file_name))
            width, height, channels = image.shape
            # At most 20% of the image width or height is used as offset
            x_offset = random.randint(0, width / 5)
            y_offset = random.randint(0, height / 5)
            # And width and height are between 80 and 100% of original
            crop_width = random.randint((4 * width) / 5, width)
            crop_height = random.randint((4 * height) / 5, height)
            # If we have 20% offset plus 100% width or height that's not gonna work
            crop_width = min(crop_width, width - x_offset)
            crop_height = min(crop_height, height - y_offset)

            print("Cropping image {} at ({}, {}) to ({}, {}).".format(image_file_name, x_offset, y_offset, crop_width, crop_height))

            # Crop image
            cropped_image = image[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]
            image_file_base_name, extension = os.path.splitext(image_file_name)
            augmented_file_name = image_file_base_name + "_cropped" + extension
            augmented_file_path = os.path.join(output_path, augmented_file_name)
            cv2.imwrite(augmented_file_path, cropped_image)

            # Adjust the annotations file

            for key in json_data:
                if image_file_name in key:
                    # We found the currently process image, now adjust bounding boxes and set the new sizes
                    image_data = json_data[key]
                    for region_key in image_data["regions"]:
                        region_data = image_data["regions"][region_key]
                        shape_attributes = region_data["shape_attributes"]
                        x, y, width, height = shape_attributes["x"], shape_attributes["y"], \
                                              shape_attributes["width"], shape_attributes["height"]
                        # max function because subtraction could be negative, then the bounding box starts at 0
                        x = max(x - x_offset, 0)
                        y = max(y - y_offset, 0)
                        # min function to account for and old position which will be cropped
                        width = width + min(x - x_offset, 0)
                        width = min(width, crop_width - x)
                        height = height + min(y - y_offset, 0)
                        height = min(height, crop_height - y)
                        shape_attributes["x"] = x
                        shape_attributes["y"] = y
                        shape_attributes["width"] = width
                        shape_attributes["height"] = height

                    # Now adjust to the new file name
                    split = key.split(image_file_name)
                    statinfo = os.stat(augmented_file_path)
                    new_key = augmented_file_name + str(statinfo.st_size)
                    image_data = json_data[key]
                    if not image_data is None:
                        # In case that we read an image that is not in the dict we skip it
                        image_data["filename"] = augmented_file_name
                        image_data["size"] = int(statinfo.st_size)
                        json_data[new_key] = json_data.pop(key)

        json.dump(json_data, modified_json_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Loads the images at the specified location, augments them and stores them at the desired location.")
    parser.add_argument("--imagesPath",
                        "-i", 
                        required=True, 
                        metavar="/path/to/images",
                        help="The path to the images.")
    parser.add_argument("--annotationsPath",
                        "-a",
                        required=True,
                        help="The path to the annotations file belonging to the images. The file has to be in the VIA format.")
    parser.add_argument("--outputPath",
                        "-o",
                        required=True,
                        help="The path where to store the augmented results.")
    parser.add_argument("--noiseType",
                        "-t",
                        required=False,
                        help="The type of the noise to apply.")
    parser.add_argument("--param1",
                        required=False,
                        help="The first parameter for noise augmentation. Can be e.g. mean for gaussian noise.")
    parser.add_argument("--param2",
                        required=False,
                        help="The second parameter for noise augmentation. Can be e.g. variance for guassian noise.")

    args = parser.parse_args()

    if args.noiseType:
        add_noise_to_images(args.imagesPath, args.annotationsPath, args.outputPath, args.noiseType, args.param1, args.param2)
    else:
        crop_and_resize_images(args.imagesPath, args.annotationsPath, args.outputPath)
