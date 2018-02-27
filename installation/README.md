# Installation

To be able to run the project you have to perform the following steps:

## Anaconda

Anaconda is an easy environment manager that comes along with many packages. It's comfortable to use and share environments.
You can skip this step if you have a different environment manager and an environment already setup. But it's probably a good
idea to use the `install.txt` file that denotes all dependencies and this is only possible with Anaconda.

* Download Anaconda at https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh.
* Make the script executable chmod +x Anaconda3-5.1.0-Linux-x86_64.sh.
* Run the script without sudo. Be sure to say yes when the script asks to add `conda` to `.bashrc`.

## Setup the Anaconda environment

* Create a new environment using the `install.txt` file: `conda create -n tensorflow -f install.txt`, where `tensorflow` is the name of the environment.
* If there is an error create a new environment without the file: `coda create -n tensorflow`. You can then install missing packages later.
* Activate the environment: `source activate tensorflow`. Your command line should now have the prefix `(tensorflow)`. If you want to deactivate the environmnt simply type: `source deactivate`. The prefix `(tensorflow)` should vanish.

## Setup COCO tools

Be sure that you have activated your environment before installing the COCO tools. This keeps the installation wihtin the environment.

* Clone the MS COCO tools repository anywhere you like: `git clone https://github.com/cocodataset/cocoapi.git`.
* `cd` into `PythonAPI`.
* Run: `make`.
* Run: `python setup.py install`.

## That's it

Now, if you enable your new environment you should be able to run the python scripts of the project.