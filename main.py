import calc_save_bbs as csb

import argparse

parser = argparse.ArgumentParser(description="This script will calculate the bounding boxes for the images at the give \
  location using the Mask RCNN implementation and store the resulting bounding boxes at the desired location. The used \
  format to store the bounding boxes is yaml.")
parser.add_argument("images_path", help="The path to the images that are to be used for detection.", type=str)
parser.add_argument("output_path", help="The path where the detected bounding boxes will be saved to", type=str)
parser.add_argument("--save_images_path", "-s", help="This path can be set if the bounding boxes should be drawn in the images. \
  The resulting images will be saved at the specified location.", type=str)
parser.add_argument("--batch_size", "-b", help="Specify the batch size, if you don't want all images to be processed at once.\
    The number of images needs to be dividable by the batch size.", type=int)
args = parser.parse_args()

csb.calc_save_bbs(args.images_path, args.output_path, args.save_images_path, args.batch_size)