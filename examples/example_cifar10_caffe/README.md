CIFAR-10 Convolution Neural Network Example 
==============================================
Example is based on standard [Caffe tutorial](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html) for [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. It's a simple classifier built on convolution, pooling and dense layers for tiny images.


Quick Start
--------------

Example supports building with [MetaWare Development tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) and [ARC GNU toolchain](https://www.synopsys.com/dw/ipdir.php?ds=sw_jtag_gnu) and running with MetaWare Debuger on [nSim simulator](https://www.synopsys.com/dw/ipdir.php?ds=sim_nSIM).

### Build with MetaWare Development tools

    Build requirements:
        - MetaWare Development tools version 2021.03 or higher
        - gmake
        - CMake version 3.18 or higher

Here we will consider building for [/hw/em9d.tcf](/hw/em9d.tcf) template. This template is a default template for this example. Other templated can be also used. 

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](/README.md#building-and-quick-start).

1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Clean previous build artifacts (optional)

       gmake clean
    or completely clean artifacts for all targets (optional)

       gmake cleanall 

3. Build example 

       gmake TCF_FILE=../../hw/em9d.tcf

4. Run example with MetaWare Debuger on nSim simulator
        
        gmake run TCF_FILE=../../hw/em9d.tcf

        Result Quality will be "S/N=8633.1     (78.7 db)"

### Build for native (x86) target:

    Build requirements:
        - Visual Studio 2019 or higher
        - gmake
        - CMake version 3.18 or higher


Build tested on Visual Studio 2019 and OS Windows 10.

1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Clean previous build artifacts (optional)

       gmake clean
    or completely clean artifacts for all targets (optional)

       gmake cleanall 

3. Build example 

       gmake ROUND_MODE=UP

4. Run example on host machine

       gmake run

        Result Quality will be "S/N=8633.1     (78.7 db)"

<!-- ### Build with ARC GNU toolchain

TEMPORARY NOTE: Currently build using ARC GNU toolchain is not supported and will be avaliable in later updates.

Here we will consider building with ARC GNU toolchain. As a platform for the assembly, we use the [IoT Devkit](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_osp/doc/build/html/board/iotdk.html) from [the embARC Open Software Platform (OSP)](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_osp/doc/build/html/introduction/introduction.html)

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

    Result Quality shall be "S/N=4383.8     (72.8 db)"

    Notes: Example built by ARC GNU tools is run using mdb_com_gnu script file. Modify this file to customize the example run mode. See [More Options on Building and Running](README.md#more-options-on-building-and-running)

### Run example without MetaWare Development tools

See documentation on [IoT Devkit](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_osp/doc/build/html/board/iotdk.html) on how to run executable built with [ARC GNU](https://foss-for-synopsys-dwc-arc-processors.github.io/toolchain/index.html) and [ARC open source development tools](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_osp/doc/build/html/index.html) on IoT Devkit.


Example Structure
--------------------
Structure of example application may be logically divided on three parts:

* **Application.** Implements Input/output data flow and data processing by the other modules. Application includes
   * ml_api_cifar10_caffe_main.c
   * ../auxiliary/examples_aux.h(.c)
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model related constants are pre-defined and model coefficients is declared in the separate compile unit 
   * cifar10_model.h
   * cifar10_model_hwcn.c
   * cifar10_constants.h
   * cifar10_coefficients_hwcn.c
* **Auxiliary code.** Various helper functions for measurements, IDX file IO, etc.
   * ../auxiliary/tensor_transform.h(.c)
   * ../auxiliary/tests_aux.h(.c)
   * ../auxiliary/idx_file.h(.c)

Example structure contains test set including small subset of CIFAR-10 (20 vectors organized in IDX file format).

Example structure also contains auxiliary files for development tools:
 * arcem9d.lcf - linkscript file for MetaWare linker.
 <!-- * memory.x    - linkscript file for GNU linker. -->
 <!-- * mdb_com_gnu - command script file for MetaWare Debugger. -->

More Options on Building and Running
---------------------------------------
Coefficients for trained NN model are stored in the separate compile unit (cifar10_coefficients_hwcn_small.c) as wrapped float numbers or integer quantized numbers in case of SA8. This allows to transform coefficients into quantized fixed point values in compile time.
For this reason you can build and check application with 8 and 16 bit depth of NN coefficients and data.

* 16 bit depth of coefficients and data (FX16) (default):
 
       gmake TCF_FILE=../../hw/em9d.tcf EXT_CFLAGS="-DMODEL_BIT_DEPTH=16"

* 8 bit depth of coefficients and data (SA8):

       gmake TCF_FILE=../../hw/em9d.tcf EXT_CFLAGS="-DMODEL_BIT_DEPTH=8"

* 8x16: 8 bit depth of coefficients and 16 bit depth of data (FX8 weights and FX16 data):

       gmake TCF_FILE=../../hw/em9d.tcf EXT_CFLAGS="-DMODEL_BIT_DEPTH=816"

Example application may be used in three modes:
1. **Built-in input processing.** Uses only hard-coded vector for the single input model inference. 
No application input arguments.

       gmake run TCF_FILE=../../hw/em9d.tcf

2. **External test-set processing.** Reads vectors from input IDX file, passes it to the model, and writes it's output to the other IDX file (if input is *tests.idx* then output will be *tests.idx_out*). 
Input test-set path is required as argument

       gmake run TCF_FILE=../../hw/em9d.tcf RUN_ARGS="small_test_base/tests.idx"

3. **Accuracy measurement for testset.** Reads vectors from input IDX file, passes it to the model, and accumulates number of successive classifications according to labels IDX file. 
Input test-set and labels paths are required as argument.

       gmake run TCF_FILE=../../hw/em9d.tcf RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"

<!-- Notes: If the example is compiled with GNU tools, then these modes are transferred to the application using mdb_com_gnu command script file for MetaWare Debugger.
       Modify this file to customize the example run mode. -->

Data Memory Requirements
----------------------------

Example application uses statically allocated memory for model weights and intermediate results (activations) and structures. Requirements for them depends on model bit depth 
configuration define and listed in table below. Before compiling application for desired hardware configuration, be sure it has enough memory to keep data.

|                      Data                              |   MODEL_BIT_DEPTH=8   |  MODEL_BIT_DEPTH=816  |  MODEL_BIT_DEPTH=16  |
| :----------------------------------------------------: | :-------------------: | :-------------------: | :------------------: |
| Weights <br/>*.mli_model* and *mli_model_p2* sections  |  33212 bytes          | 33212 bytes           | 66420 bytes          |
| Activations 1 <br/>*.Zdata* section                    |  32768 bytes          | 65536 bytes           | 65536 bytes          |
| Activations 2 <br/>*.Ydata* section                    |  8192 bytes           | 16384 bytes           | 16384 bytes          |

By default, application uses MODEL_BIT_DEPTH=16 mode. Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

References
----------------------------
CIFAR-10 Dataset:
> Alex Krizhevsky. *"Learning Multiple Layers of Features from Tiny Images."* 2009.

Caffe framework:
> Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor. *"Caffe: Convolu-tional Architecture for Fast Feature Embedding."* arXiv preprint arXiv:1408.5093. 2014: http://caffe.berkeleyvision.org/

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
