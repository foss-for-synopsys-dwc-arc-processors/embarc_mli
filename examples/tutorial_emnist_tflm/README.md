EMNIST Conversion Example
========================================================================
This example shows how to convert EMNIST Tensorflow model into Tensorflow Lite Micro format and use it in embARC MLI application.

**Important note:** this example won't work with EM/HS platform. For EM/HS, please use [MLI 1.1 release version](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_1.1/examples/tutorial_emnist_tflm).

## Requirements
---------------
* MetaWare Development Tools
    * [Order the ARC MetaWare Development Toolkit](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware)
    * [Evaluation license](https://eval.synopsys.com/)
* gmake (pre-installed as a part of MetaWare tools)
* embARC MLI Library
    * `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli`
* Tensorflow Lite for Microcontrollers (part of Tensorflow)
    * `git clone https://github.com/tensorflow/tflite-micro.git`
* Text Editor

Installation process of the following dependencies is described in [Getting Started](#getting-started) section:
* Python 3.7
* virtualenv (optional)
* Dependencies from requirements.txt
    * NumPy 1.19.2
    * Matplotlib
    * Jupyter Lab / Notebook
    * TensorFlow 2.5.0
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
pip install -r ./conversion_tutorial/requirements.txt
python -c "import emnist; emnist.ensure_cached_data();"
```
## Generate Tensorflow Lite Micro library
Open root directory of tensorflow in terminal (use Cygwin or MinGW terminal if you're on Windows). Run:
```bash
make -f tensorflow/lite/micro/tools/make/Makefile\ 
OPTIMIZED_KERNEL_DIR=arc_mli TARGET=arc_custom\ 
TCF_FILE=<path_to_vpx_tcf_file>\ 
ARC_TAGS=mli20_experimental microlite
```
Generated library *libtensorflow-microlite.a* can be found in *\{tensorflow-dir\}/tensorflow/lite/micro/tools/make/gen/\{target\}/lib*. Copy it to third_party directory of this example.

## Convert the model
To convert the model, run the Jupyter Notebook:
```bash
jupyter notebook conversion_tutorial/model_conversion.ipynb
```
After completing the tutorial you should have model and test samples generated. 

Please make sure you don't forget to use [Model Adaptation Tool](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro/tree/arc_mli_20_temp/tensorflow/lite/micro/tools/make/targets/arc#model-adaptation-tool-experimental-feature) to convert `emnist_model_int8.tflite` you got before saving it as C array.

Copy *conversion_tutorial/generated/model.h* and *conversion_tutorial/generated/test_samples.cc* to *src* folder.

Before building you should set TENSORFLOW_DIR variable to point to your Tensorflow top folder. E.g. on Windows:
```bash
set TENSORFLOW_DIR=\{your-path-to-tflite-micro\}
```

Now you can build the example using MetaWare Development Tools:
```bash
gmake app TCF_FILE=<path_to_vpx_tcf_file>
```
And run the example with DesignWare ARC nSIM simulator:
```bash
gmake run TCF_FILE=<path_to_vpx_tcf_file>
```

