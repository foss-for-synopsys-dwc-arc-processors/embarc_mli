EMNIST Convolution Neural Network Example
========================================================================
This example shows how to convert EMNIST Tensorflow model into Tensorflow Lite Micro format and use it in embARC MLI application. 

## Requirements
---------------
* MetaWare Development Tools
    * [Order the ARC MetaWare Development Toolkit](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware)
    * [Evaluation license](https://eval.synopsys.com/)
* gmake (pre-installed as a part of MetaWare tools)
* embARC MLI Library
    * `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli`
* Tensorflow Lite for Microcontrollers (part of Tensorflow)
    * `git clone https://github.com/tensorflow/tensorflow.git`
* Text Editor

Installation process of the following dependencies is described in [Getting Started](#getting-started) section:
* Python 3.7
* virtualenv (optional)
* Dependencies from requirements.txt
    * NumPy 1.16.4
    * Matplotlib
    * Jupyter Lab / Notebook
    * tf-nightly 2.3
    * Keras
    * emnist 
    
## Getting started
---------------

## Install Python and create a virtual environment

0. It is recommended that you uninstall your previous Python distribution.
1. Download official [Python 3.7 distribution](https://www.python.org/ftp/python/3.7.4/python-3.7.4-amd64.exe).
2. Install py launcher and pip. Do not add Python to the PATH. After this, the `py` command in command line is your entry point to Python interpreter.
3. (optional) Install virtualenv with `py -m pip install --upgrade pip virtualenv`
4. (optional) Create virtual environment with `py -m virtualenv py_env`.
5. (optional) Activate virtual environment with `./py_env/Scripts/activate`.
6. Execute ` cd ./embarc_mli/examples/tutorial_emnist_tflm`.

## Install pip requirements
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
python -c "import emnist; emnist.ensure_cached_data();
```
## Generate Tensorflow Lite Micro library
Open root directory of tensorflow in terminal (use Cygwin or MinGW terminal if you're on Windows). Run:
```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET_ARCH=arc microlite
```
Generated library *libtensorflow-microlite.a* can be found in *<tensorflow-dir>/tensorflow/lite/micro/tools/make/gen/<target>/lib*. Copy it to third_party directory of this example.

## Convert the model
To convert the model, run the Jupyter Notebook:
```bash
jupyter notebook conversion_tutorial/model_conversion.ipynb
```
After completing the tutorial you should have model and test samples generated. Copy *conversion_tutorial/generated/model.h* and *conversion_tutorial/generated/test_samples.cc* to *src* folder.

Now you can build and run the example with DesignWare ARC nSIM simulator:
```bash
gmake app run 
```

