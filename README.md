# forschungsprojekt

# Setup

The following is supposed to describe the setup process to be able to run the Mask RCNN network (https://github.com/matterport/Mask_RCNN).

* Download and install Anaconda (https://conda.io/docs/user-guide/install/download.html)
* Create a virtual environment with Python 3.5 
```
conda create -n tensorflow python=3.5
```
* Activate the environment: source activate tensorflow
* Do the following with activated source
```
conda install tensorflow
```
```
conda install matplotlib
```
```
conda install scikit-image
```
```
conda install opencv
```
```
conda install cython
```
* Clone with git: https://github.com/cocodataset/cocoapi.git.
* cd into the cocoapi/PythonAPI folder
* execute 
```
make
```
* Still in the folder execute
```
python setup.py install
```
* Some more steps necessary with Cython etc.

* Clone with git: https://github.com/matterport/Mask_RCNN **Mask RCNN's code has to be on the same level as this project's code, i.e. the folder Mask_RCNN and forschungsprojekt have to be in the same folder, as forschungsprojekt uses a relative path to Mask RCNN.**
* Download the weights for the network at https://github.com/matterport/Mask_RCNN/releases and put them in the source folder of Mask RCNN
