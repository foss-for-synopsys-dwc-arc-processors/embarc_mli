CIFAR-10 Convolution Neural Network Example 
==============================================
Example is based on standard [Caffe tutorial](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html) for [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. It's a simple classifier built on convolution, pooling and dense layers for tiny images.


Quick Start
--------------

Example supports building using MetaWare Development tools and ARC GNU toolchain and running with MetaWare Debuger on nSim simulator.

### Build with MetaWare Development tools

    Build requirements:
        - MetaWare Development tools version 2018.12 or higher
        - gmake

Here we will consider building for [/hw/em9d.tcf](/hw/em9d.tcf) template. This template is a default template for this example. Other templated can be also used. 

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](/README.md#building-and-quick-start).

1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Clean previous build artifacts (optional)

       gmake clean

3. Build example 

       gmake TCF_FILE=../../hw/em9d.tcf 

### Run example with MetaWare Debuger on nSim simulator.

       gmake run TCF_FILE=../../hw/em9d.tcf

    Result Quality shall be "S/N=3638.6     (71.2 db)"

### Build with ARC GNU toolchain

Here we will consider building with ARC GNU toolchain. As a platform for the assembly, we use the [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) from [the embARC Open Software Platform (OSP)](https://embarc.org/embarc_osp/doc/build/html/introduction/introduction.html#)

    Build requirements:
        - ARC GNU toolchain version 2018.09 or higher
        - embARC MLI Library prebuilt with MetaWare Development tools for IoT Devkit hardware configuration
        - gmake

0. Prebuilt embARC MLI Library  must be copyied into the ./examples/prebuilt folder.

1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Clean previous build artifacts (optional)

        gmake TOOLCHAIN=gnu clean

3. Build example

        gmake TOOLCHAIN=gnu

   Notes: IoT Devkit hardware configuration is specifed in Makefile. Additionally used memory.x linkscript file for GNU linker. 

### Run example with MetaWare Debuger on nSim simulator.

    Run requirements:
    - MetaWare Development tools version 2018.12 or higher
    - arcem9d.tcf file with hardware configuration of IoT Devkit for setup nSim.

0. Copy the [arcem9d.tcf](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_osp/blob/master/board/iotdk/configs/10/tcf/arcem9d.tcf) file into example folder.

1. Run example 

        gmake run TOOLCHAIN=gnu TCF_FILE=arcem9d.tcf

    Result Quality shall be "S/N=3638.6     (71.2 db)"

    Notes: Example built by ARC GNU tools is run using mdb_com_gnu script file. Modify this file to customize the example run mode. See [More Options on Building and Running](README.md#more-options-on-building-and-running)

### Run example without MetaWare Development tools

See documentation on [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) on how to run executable built with [ARC GNU](https://embarc.org/toolchain/index.html) and [ARC open source development tools](https://embarc.org/embarc_osp/doc/build/html/index.html) on IoT Devkit.


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

Example structure contains test set including small subset of CIFAR-10 (20 vectors organized in IDX file format).

Example structure also contains auxiliary files for development tools:
 * arcem9d.lcf - linkscript file for MetaWare linker.
 * memory.x    - linkscript file for GNU linker.
 * mdb_com_gnu - command script file for MetaWare Debugger.

More Options on Building and Running
---------------------------------------
CIFAR-10 example application is implemented in the same way as LSTM Based HAR example and provides the same configuration and running abilities. For more details see appropriate HAR example [description part](/examples/example_har_smartphone/README.md#more-options-on-building-and-running).

References
----------------------------
CIFAR-10 Dataset:
> Alex Krizhevsky. *"Learning Multiple Layers of Features from Tiny Images."* 2009.

Caffe framework:
> Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor. *"Caffe: Convolu-tional Architecture for Fast Feature Embedding."* arXiv preprint arXiv:1408.5093. 2014: http://caffe.berkeleyvision.org/

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]



