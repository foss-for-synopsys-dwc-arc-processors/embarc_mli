LSTM Based Human Activity Recognition (HAR) Example  
==============================================
Example shows how to work with recurrent primitives (LSTM and basic RNN) implemented in embARC MLI Library. It is based on open source [GitHub project](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) by Guillaume Chevalie. Chosen approach, complexity of the model and [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) are relevant to IoT domain. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone. Classes:
 * 0: WALKING
 * 1: WALKING_UPSTAIRS
 * 2: WALKING_DOWNSTAIRS
 * 3: SITTING
 * 4: STANDING
 * 5: LAYING

Quick Start
--------------

Example supports building using MetaWare Development tools and ARC GNU toolchain and running with MetaWare Debuger on nSim simulator.

### Build with MetaWare Development tools

    Build requirements:
        - MetaWare Development tools version 2018.12 or higher
        - gmake

Here we will consider building for [/hw/em9d.tcf](/hw/em9d.tcf) template. This template is a default template for this example. Other templated can be also used. 

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](/README.md).

1. Open command line and change working directory to './examples/example_har_smartphone'

2. Clean previous build artifacts (optional)

       gmake clean

3. Build example 

       gmake TCF_FILE=../../hw/em9d.tcf 

### Run example with MetaWare Debuger on nSim simulator.

       gmake run TCF_FILE=../../hw/em9d.tcf

    Result Quality shall be "S/N=1823.9     (65.2 db)"

### Build with ARC GNU toolchain

Here we will consider building with ARC GNU toolchain. As a platform for the assembly, we use the [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) from [the embARC Open Software Platform (OSP)](https://embarc.org/embarc_osp/doc/build/html/introduction/introduction.html#)

    Build requirements:
        - ARC GNU toolchain version 2018.09 or higher
        - embARC MLI Library prebuilt with MetaWare Development tools for IoT Devkit hardware configuration
        - gmake

0. Prebuilt embARC MLI Library  must be copyied into the /prebuilt folder.

1. Open command line and change working directory to './examples/example_har_smartphone'

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

    Result Quality shall be "S/N=1823.9     (65.2 db)"

    Notes: Example built by ARC GNU tools is run using mdb_com_gnu script file. Modify this file to customize the example run mode. See [More Options on Building and Running](README.md#more-options-on-building-and-running)

### Run example without MetaWare Development tools

See documentation on [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) on how to run executable built with [ARC GNU](https://embarc.org/toolchain/index.html) and [ARC open source development tools](https://embarc.org/embarc_osp/doc/build/html/index.html) on IoT Devkit.


Example Structure
--------------------
Structure of example application may be divided logically on three parts:

* **Application.** Implements Input/output data flow and it's processing by the other modules. Application includes:
   * ml_api_har_smartphone_main.c
   * ../auxiliary/examples_aux.h(.c)
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model related constants are pre-defined and model coefficients is declared in the separate compile unit 
   * har_smartphone_model.h
   * har_smartphone_model.c
   * har_smartphone_constants.h
   * har_smartphone_coefficients.c
* **Auxiliary code.** Various helper functions for measurements, IDX file IO, etc.
   * ../auxiliary/tensor_transform.h(.c)
   * ../auxiliary/tests_aux.h(.c)
   * ../auxiliary/idx_file.h(.c)

Example structure contains test vector set including small subset of pre-processed UCI HAR Smartphones dataset (20 vectors organized in IDX file format).

Example structure also contains auxiliary files for development tools:
 * arcem9d.lcf - linkscript file for MetaWare linker.
 * memory.x    - linkscript file for GNU linker.
 * mdb_com_gnu - command script file for MetaWare Debugger.

More Options on Building and Running
---------------------------------------
Coefficients for trained NN model are stored in the separate compile unit (*coefficients.c) as wrapped float numbers. This allows to transform coefficients into quantized fixed point values in compile time.
For this reason you can build and check application with 8 and 16 bit depth of NN coefficients and data.

* 16 bit depth of coefficients and data (default):
 
       gmake TCF_FILE=../../hw/em9d.tcf EXT_CFLAGS="-DMODEL_BIT_DEPTH=16"

* 8 bit depth of coefficients and data:

       gmake TCF_FILE=../../hw/em9d.tcf EXT_CFLAGS="-DMODEL_BIT_DEPTH=8"

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

Notes: If the example is compiled with GNU tools, then these modes are transferred to the application using mdb_com_gnu command script file for MetaWare Debugger.
       Modify this file to customize the example run mode.

References
----------------------------
GitHub project served as starting point for this example:
> Guillaume Chevalier, *LSTMs for Human Activity Recognition*, 2016,[https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

Human Activity Recognition Using Smartphones [Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones):
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *"A Public Domain Dataset for Human Activity Recognition Using Smartphones."* 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013:

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
