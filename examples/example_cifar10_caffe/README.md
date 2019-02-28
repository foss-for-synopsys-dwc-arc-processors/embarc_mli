CIFAR-10 Convolution Neural Network Example 
==============================================
Example is based on standard [Caffe tutorial](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html) for [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. It's a simple classifier built on convolution, pooling and dense layers for tiny images.


Quick Start
--------------
hw/em9d.tcf template is a default template for this example. Other templated can be also used. 

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](public/README.md).

1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Clean previous build artifacts (optional)

       gmake clean

3. Build example 

       gmake TCF_FILE=../../hw/em9d.tcf 

4. Run example 

       gmake run TCF_FILE=../../hw/em9d.tcf

5. Result Quality shall be "S/N=2344.4     (67.4 db)"


Example Structure
--------------------
Structure of example application may be logically divided on three parts:

* **Application.** Implements Input/output data flow and data processing by the other modules. Application includes
   * ml_api_cifar10_caffe_main.c
   * ../auxiliary/examples_aux.h(.c)
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model related constants are pre-defined and model coefficients is declared in the separate compile unit 
   * cifar10_model.h
   * cifar10_model_chw.c (cifar10_model_hwc.c)
   * cifar10_constants.h
   * cifar10_coefficients_chw.c (cifar10_coefficients_hwc.c)
* **Auxiliary code.** Various helper functions for measurements, IDX file IO, etc.
   * ../auxiliary/tensor_transform.h(.c)
   * ../auxiliary/tests_aux.h(.c)
   * ../auxiliary/idx_file.h(.c)

Example structure also contains test set including small subset of CIFAR-10 (20 vectors organized in IDX file format).
   
More Options on Building and Running
---------------------------------------
CIFAR-10 example application is implemented in the same way as LSTM Based HAR example and provides the same configuration and running abilities. For more details see appropriate HAR example [description part](public/examples/example_har_smartphone/README.md).

References
----------------------------
CIFAR-10 Dataset:
> Alex Krizhevsky. *"Learning Multiple Layers of Features from Tiny Images."* 2009.

Caffe framework:
> Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor. *"Caffe: Convolu-tional Architecture for Fast Feature Embedding."* arXiv preprint arXiv:1408.5093. 2014: http://caffe.berkeleyvision.org/

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]



