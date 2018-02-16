import os
import h5py
import numpy as np
import misc

def adjust_weights(weightsPath, weightsOutputPath):
	""" This function extracts the weights at the given path and omits all information related
	to COCO classes that are not present in VUFO. Beware that it simply transfers the first connections
	until the number of VUFO classes is reached, without considering that a VUFO class might not
	have the same index in the COCO classes.
	"""

	assert os.path.exists(weightsPath)
	#assert not os.path.exists(weightsOutputPath)

	# Those are the last layers where we have to omit weights related to the non-VUFO classes
	lastLayers = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"]
	numCocoClasses = len(misc.COCO_CLASSES)
	numVufoClasses = len(misc.VUFO_CLASSES)

	with h5py.File(weightsOutputPath, "w") as weightsOutputFile:
		with h5py.File(weightsPath, "r") as weightsFile:
			for layer in weightsFile.keys():
				layerData = weightsFile[layer]
				if layer in lastLayers:
					group = weightsOutputFile.create_group(layer + "/" + layer)

					# The layer is within the specified layers, i.e. we have to 
					# check the indices of the table before transfering the weights

					# The dictionary is nested for some reason
					nestedLayerData = layerData[layer]

					newBias = None
					newKernel = None

					if layer == "mrcnn_bbox_fc":
						# Layer dimension is 324 = 4 * num(COCO_CLASSES)
						bias = nestedLayerData["bias:0"]
						numberOfConnectionsToKeep = int(len(bias) / numCocoClasses) * numVufoClasses
						newBias = bias[:numberOfConnectionsToKeep]
						newKernel = nestedLayerData["kernel:0"][:numberOfConnectionsToKeep][:]
					elif layer == "mrcnn_class_logits":
						newBias = nestedLayerData["bias:0"][:numVufoClasses]
						newKernel = nestedLayerData["kernel:0"][numVufoClasses:][:]
					elif layer == "mrcnn_mask":
						newBias = nestedLayerData["bias:0"][:numVufoClasses]
						# Kernel dimension of mask layer is 1 x 1 x 256 x 81
						newKernel = []
						newKernel.append([])
						newKernel[0][0] = nestedLayerData["kernel:0"][:][:][:]
						print(newKernel.shape[3])

					group.create_dataset("bias:0", data=newBias)
					group.create_dataset("kernel:0", data=newKernel)
				else:
					groupId = weightsOutputFile.require_group(layer)
					weightsFile.copy(layer, groupId)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Loads the specified weights and stores them at the specified location. During " +
    	"this process the number of outputs is adjusted to the number of VUFO classes to omit non-VUFO COCO classes. The number " +
    	"of classes can be seen in misc.py.")
    parser.add_argument("--weightsPath",
                        "-w", 
                        required=True, 
                        metavar="/path/to/weights.h5",
                        help="The path to the weights.")
    parser.add_argument("--outputPath",
    					"-o",
    					required=True,
    					metavar="/path/to/output/file.h5")

    args = parser.parse_args()
    adjust_weights(args.weightsPath, args.outputPath)