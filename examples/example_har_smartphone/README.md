LSTM Based Human Activity Recognition (HAR) Example 
==============================================
Example shows how to work with recurrent primitives (LSTM and basic RNN) implemented in embARC MLI Library. It is based on open source GitHub project by Guillaume Chevalie. Chosen approach, complexity of the model and dataset are relevant to IoT domain. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone. Classes:

 * 0: WALKING
 * 1: WALKING_UPSTAIRS
 * 2: WALKING_DOWNSTAIRS
 * 3: SITTING
 * 4: STANDING
 * 5: LAYING

# Directory Structure

* **Application**
  * `ml_api_har_smartphone_main.c`                 		- implements input/output data flow and its processing by the other modules.
* **Inference Module**
  * `har_smartphone_model[_sa|_fx].h(.c)`                 		- implement the model using embARC MLI Library according to a pre-defined graph separately for each quantization type.
  * `har_smartphone_constants.h`                 		- contains model and tensor configuration.
  * `har_smartphone_coefficients[_sa|_fx].c`                 		- contain tensor coefficients and quantization parameters separately for each quantization type.
* **Datasets**
  * `ir_idx_300`                 		- contains output tensors of each layer for a single input processing case.
  * `small_test_base`                 		- contains a small dataset for accuracy measurment.
* **Auxiliary files**
  * `arcem9d.lcf`                 		- linkscript file for MetaWare linker.

# Building and Running

## Building with MetaWare Development Tools

Example supports building with [MetaWare Development tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) and running with MetaWare Debuger on [nSim simulator](https://www.synopsys.com/dw/ipdir.php?ds=sim_nSIM). 

[General build requirements and instructions](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli#general-build-process)

Build artifacts of the application are stored in the `/obj/<project>/examples/example_har_smartphone` directory where `<project>` is defined according to your target platform.  

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](/README.md#building-and-quick-start).

1. Open command line and change working directory to './examples/example_har_smartphone'

2. Clean previous build artifacts (optional)

       gmake clean
    or completely clean artifacts for all targets (optional)

       gmake cleanall
 
3. Follow [general instructions](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli#build-command-examples-for-arc-processors) on TCF generation and build process 

4. Run example with MetaWare Debuger on nSim simulator

       gmake run TCF_FILE=./hw/vpx5_integer_full.tcf

       Result Quality will be "S/N=1754.9     (64.9 db)"

## Building for native (x86) target

[Build requirements](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli#x86-host-emulation)


Build tested on Visual Studio 2019 and OS Windows 10.

1. Open command line and change working directory to './examples/example_har_smartphone/'

2. Clean previous build artifacts (optional)

       gmake clean
    or completely clean artifacts for all targets (optional)

       gmake cleanall 

3. Build example 

       gmake ROUND_MODE=UP

4. Run example on host machine

       gmake run

       Result Quality will be "S/N=1754.9     (64.9 db)"
## HAR Example Specific Extra Options. 

Coefficients for trained NN model are stored in the separate compile unit (*coefficients[_sa|_fx].c). You can build and check application with 8 and 16 bit depth of NN coefficients and data.

* 16 bit depth of coefficients and data (default):
 
       gmake run_FX16 TCF_FILE=../../hw/em9d.tcf

* 8 bit depth of coefficients and data (SA8):

       gmake run_SA8 TCF_FILE=../../hw/em9d.tcf

* 8x16: 8 bit depth of coefficients and 16 bit depth of data (FX8 weights and FX16 data):

       gmake run_FX16_FX8_FX8 TCF_FILE=../../hw/em9d.tcf

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

# Data Memory Requirements

Example application uses statically allocated memory for model weights and intermediate results (activations) and structures. Requirements for them depends on model bit depth 
configuration define and listed in table below. Before compiling application for desired hardware configuration, be sure it has enough memory to keep data.

|                      Data                         |   MODEL_BIT_DEPTH=8   |  MODEL_BIT_DEPTH=816  |  MODEL_BIT_DEPTH=16  |
| :-----------------------------------------------: | :-------------------: | :-------------------: | :------------------: |
| Weights <br/>(*.mli_model* section)              |  18040 bytes          | 17160 bytes           | 34316 bytes          |
| Activations <br/>(*.Xdata* and *.Ydata* sections) |  29824 bytes           | 26944 bytes           | 26944 bytes          |
| Structures <br/>(*.mli_data* section)           |  1540 bytes            | 1220 bytes             | 1220 bytes            |

By default, application uses MODEL_BIT_DEPTH=16 mode. Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

# References

GitHub project served as starting point for this example:
> Guillaume Chevalier, *LSTMs for Human Activity Recognition*, 2016,[https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

Human Activity Recognition Using Smartphones [Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones):
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *"A Public Domain Dataset for Human Activity Recognition Using Smartphones."* 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013:

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
