Model Deployment Tutorial for Caffe and CIFAR-10: Scripts
==================================================

This folder contains python scripts to reflect deployment process described in the 
[heading tutorial](https://embarc.org/embarc_mli/doc/build/html/Examples_Tutorials/Examples_Tutorials.html#model-deployment-tutorial-for-caffe-and-cifar-10).

### Requirements

- Python (tested for 3.7 and 2.7)
- Caffe (Python module must be installed and available in environment)
- lmdb
- numpy

### Quick start

Main entry point is ``deployment_main.py`` script. To print available options run:

    python deployment_main.py --help

Caffe standard Cifar-10 tutorial provides tool for CIFAR-10 dataset transformation into LMDB form. User must provide paths to these folders for proper inference time data definition.

    python deployment_main.py --lmdb_data_dir=<Path>

### Structure

``README.md``                    - This File  
``cifar10_small.prototxt``     - Structure of trained model  
``cifar10_small.caffemodel.h5``  - Caffe Model with coefficients trained on CIFAR-10 dataset  
``deployment_main.py``           - Main script  
``deployment_steps.py``          - Deployment steps implemented in a set of functions  
``mli_fxtools.py``               - Helper classes for accounting statistics and quantization  


### Details

For more information see:
1. [Model Deployment Tutorial for Caffe and CIFAR-10](https://embarc.org/embarc_mli/doc/build/html/Examples_Tutorials/Examples_Tutorials.html#model-deployment-tutorial-for-caffe-and-cifar-10)
2. Python code which is well documented

