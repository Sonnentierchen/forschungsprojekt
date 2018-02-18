import cv2
import os
import shutil
import numpy as np
import json
import misc

def add_noise_to_images(images_path, annotations_path, output_path, noise_type, param_1, param_2):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_files_names = misc.get_images_at_path(images_path)

    annotations_file_name = os.path.basename(annotations_path)
    shutil.copy(annotations_path, os.path.join(output_path, annotations_file_name))
   
    for image_file_name in image_files_names:
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

        if not noisy is None:
            cv2.imwrite(os.path.join(output_path, image_file_name), noisy)

def crop_and_resize_images(images_path, annotations_path, output_path):
    image_files_names = misc.get_images_at_path(images_path)

    annotations_file_name = os.path.basename(annotations_path)
    shutil.copy(annotations_path, os.path.join(output_path, annotations_file_name))
   
    #for image_file_name in image_files_names:
    # TODO

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
                        help="The path to the annotations file belonging to the images.")
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
