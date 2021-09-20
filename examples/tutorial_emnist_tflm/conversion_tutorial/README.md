# Model Conversion Tutorial for EMNIST Example

This tutorial shows how to convert EMNIST Tensorflow model into Tensorflow Lite Micro format for further usage in application together with embARC MLI.

## Requirements

The following dependencies must be installed in your environment:

* Python 3.7
* Dependencies from the [requirements.txt](/examples/tutorial_emnist_tflm/conversion_tutorial/requirements.txt) file
  * NumPy 1.19.2
  * Matplotlib
  * Jupyter Lab / Notebook
  * TensorFlow 2.5.1
  * Keras
  * emnist

If you are looking for instruction to set-up a compatible environment, then consider the comprehensive guide on [TensorFlow](https://www.tensorflow.org/install) installation. We recommend to follow ["Install TensorFlow with pip"](https://www.tensorflow.org/install/pip) and not ignore recommendation on usage of the virtual environment.

Make sure you have installed required version of dependencies listed in the *requirements.txt* file. In case you followed our recommendation on install TensorFlow with pip, for [step 3](https://www.tensorflow.org/install/pip#3.-install-the-tensorflow-pip-package) you can use the following commands instead the proposed ones. First change working directory in terminal to the [conversion_tutorial](/examples/tutorial_emnist_tflm/conversion_tutorial).

    pip3 install -r ./requirements.txt
    python3 -c "import emnist; emnist.ensure_cached_data();"

## Running Conversion Tutorial

To convert the model, run the Jupyter Notebook (change working directory in terminal to the [conversion_tutorial](/examples/tutorial_emnist_tflm/conversion_tutorial) if you haven't done it):

    jupyter notebook model_conversion.ipynb

After completing the tutorial, you should have model and test samples generated.

Please make sure you don't forget to use [Model Adaptation Tool](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/targets/arc/adaptation_tool.py) to convert `emnist_model_int8.tflite` you got before saving it as C array.

## Using Converted Model in Application

See [EMNIST example part](/examples/tutorial_emnist_tflm) on how a converted model might be used in application together with TFLM and embARC MLI. The example already contains converted model, hence this step is optional.

To use generated models in the example you need to copy */examples/tutorial_emnist_tflm/conversion_tutorial/generated/model.h* and */examples/tutorial_emnist_tflm/conversion_tutorial/generated/test_samples.cc* to */examples/tutorial_emnist_tflm/src* folder. Make sure that names of buffers and constants in copied files are aligned with the former ones and change them if not. Afterward you can clean, build and run example application according to instructions.
