EMNIST Convolution Neural Network Example
========================================================================
This example shows the porting process of Keras EMNIST recognition Convolutional Neural Network to [embARC MLI Library](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli). Resulting output can be executed with DesignWare ARC nSIM simulator.

## Requirements
---------------
* MetaWare Development Tools 2019.06 (or later)
    * [Order the ARC MetaWare Development Toolkit](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware)
    * [Evaluation license](https://eval.synopsys.com/)
* gmake (pre-installed as a part of MetaWare tools)
* embARC MLI Library
    * `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli`
* Text Editor

Installation process of the following dependencies is described in [Getting Started](#getting-started) section:
* Python 3.7
* virtualenv (optional)
* Dependencies from requirements.txt
    * NumPy 1.16.4
    * Matplotlib
    * Jupyter Lab / Notebook
    * TensorFlow 2.0
    * Keras 2.2.4
    * emnist 
    * keras-tqdm
    * idx2numpy
    * ipywidgets 7.5.1
    
## Example Structure
---------------

* `example.ipynb` - Jupyter Notebook containing steps neccessary for model training and deployment
* `mli_cnn_bn.h5` - pre-trained Keras model for fast deployment
* `emnist_keras_deployment/` - deployment helper module for example.ipynb
* `widgets_def.py` - example.ipynb widgets description 
* `requirements.txt` - notebook requirements

## Getting started
---------------

## Install Python and create a virtual environment

0. It is recommended that you uninstall your previous Python distribution.
1. Download official [Python 3.7 distribution](https://www.python.org/ftp/python/3.7.4/python-3.7.4-amd64.exe).
2. Install py launcher and pip. Do not add Python to the PATH. After this, the `py` command in command line is your entry point to Python interpreter.
3. (optional) Install virtualenv with `py -m pip install --upgrade pip virtualenv`
4. (optional) Create virtual environment with `py -m virtualenv py_env`.
5. (optional) Activate virtual environment with `./py_env/Scripts/activate`.
6. Execute ` cd ./embarc_mli/examples/tutorial_emnist_tensorflow`.

## Install pip requirements
```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
python -c "import emnist; emnist.ensure_cached_data();
```
## Deploy and test the model
To train and deploy the model, run the Jupyter Notebook:
```bash
jupyter notebook example.ipynb
```

After completing the tutorial, you can run the example with DesignWare ARC nSIM simulator. The next line executes the model for a single hard-coded letter prediction providing the information on performance:
```bash
gmake run app
```

You can also test the model on a data subset generated during the tutorial:
```bash
gmake run RUN_ARGS="model/small_test_base/tests.idx model/small_test_base/labels.idx"
```



