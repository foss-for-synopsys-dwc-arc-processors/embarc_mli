LSTM Based Human Activity Recognition (HAR) Example 
==============================================
Example shows how to work with recurrent primitives (LSTM and basic RNN) implemented in embARC MLI Library. It is based on open source GitHub project by Guillaume Chevalie. Chosen approach, complexity of the model and dataset are relevant to IoT domain. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone. Classes:

 * 0: WALKING
 * 1: WALKING_UPSTAIRS
 * 2: WALKING_DOWNSTAIRS
 * 3: SITTING
 * 4: STANDING
 * 5: LAYING

**Important note:** Example doesn’t work for VPX configurations without guard bits as it produces incorrect results due to accumulator overflow during calculations.

# Building and Running

You need to configure and build the library project for the desired platform. 
Please read the corresponding section on [building the package](/README.md#building-the-package). 
There are no extra requirements specific for this application. All the specified platforms are supported by the test application.  

Build artifacts of the application are stored in the `/obj/<project>/examples/example_har_smartphone` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps. 
You need to replace `<options>` placeholder in commands below with the same options list you used for the library configuration and build. 

1. Open command line in the root of the embARC MLI repo and change working directory to './examples/example_har_smartphone/'

       cd ./examples/example_har_smartphone/

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build the example. This is an optional step as you may go to the next step which automatically invokes the build process. 

       gmake <options> build

4. Run the example

       gmake <options> run

## More Options on Running

LSTM Based HAR example application is implemented in the same way as  CIFAR-10 example and provides the same configuration and running abilities. For more details see appropriate [CIFAR-10 example description part](/examples/example_cifar10_caffe#more-options-on-running).


##  x86 Build Process Example

Assuming your environment satisfies all build requirements for x86 platform, you can use the following script to build. 
The first step is to open a command line and change working directory to the root of the embARC MLI repo.

1. Clean all previous artifacts for all platforms
    ```bash
    gmake cleanall 
    ```

2. Build project to emulate ARC VPX platform. Use multithreaded build process (4 threads):
    ```bash
    gmake ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 build  
    ```

3. Change working directory  and build the example:
    ```bash
    cd ./examples/example_har_smartphone
    gmake ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 build
    ```

4. Run example w/o input arguments for all supported data types:
    ```bash
    gmake ROUND_MODE=UP FULL_ACCU=OFF run 
    gmake ROUND_MODE=UP FULL_ACCU=OFF run_SA8 
    gmake ROUND_MODE=UP FULL_ACCU=OFF run_FX16_FX8_FX8
    ```

4. Run example in accuracy measurements mode using provided small test set :
    ```bash
    gmake ROUND_MODE=UP FULL_ACCU=OFF run RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"

    gmake ROUND_MODE=UP FULL_ACCU=OFF run_SA8 RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"
    ```

##  ARC VPX Build Process Example

Assuming your environment satisfies all build requirements for ARC VPX platform, you can use the following script to build and run application using the nSIM simulator.
The first step is to open a command line and change working directory to the root of the embARC MLI repo.

1. Clean all previous artifacts for all platforms
    ```bash
    gmake cleanall 
    ```

2. Generate recommended  TCF file for VPX
    ```bash
    tcfgen -o ./hw/vpx5_integer_full.tcf -tcf=vpx5_integer_full -iccm_size=0x80000 -dccm_size=0x40000
    ```

3. Build project using generated TCF and appropriate built-in runtime library for it. Use multithreaded build process (4 threads):
    ```bash
    gmake TCF_FILE=./hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full JOBS=4 build
    ```

4. Change working directory  and build the example:
    ```bash
    cd ./examples/example_har_smartphone
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full build
    ```

5. Run example w/o input arguments for all supported data types:
    ```bash
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run 

    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run_SA8 

    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run_FX16_FX8_FX8
    ```

6. Run example in accuracy measurements mode using provided small test set :
    ```bash
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run_FX16 RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"

    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run_SA8 RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"
    ```

## Expected Output

Console Output depends on the build options, chosen application mode and target run commands (application arguments).  

### 1. **Built-in input processing.**

Console output may look like: 

       HARDCODED INPUT PROCESSING
       ir_mov.idx(w/o IR check):       X cycles
       ir_in.idx(w/o IR check):        X cycles
       ir_relu1.idx(w/o IR check):     X cycles
       ir_lstm2.idx(w/o IR check):     X cycles
       ir_lstm3.idx(w/o IR check):     X cycles
       ir_fc4.idx(w/o IR check):       X cycles


       Summary:
              Movement: X cycles
              Conversion: X cycles
              Layer1: X cycles
              Layer2: X cycles
              Layer3: X cycles
              Layer4: X cycles

              Total: X cycles

       Result Quality: S/N=5346.3     (74.6 db)
       FINISHED

where:
* `X cycles` reflects number of cycles for a specific layer or in total . `X` may vary depending on target platform and build options.

* `Result Quality: S/N=5346.3     (74.6 db)` reflects the signal-to-noise ration of the model output in comparison with reference float. The ratio itself (`S/N` and `x db`) may vary depending on the target platform and `run_*` command. In particular :

  * `run_FX16`: Result may slightly fluctuates around `S/N=5346.3     (74.6 db)` 
  * `run_FX16_FX8_FX8`: Result may slightly fluctuates around `S/N=56.0       (35.0 db)` 
  * `run_SA8`: Result may slightly fluctuates around `S/N=4.6        (13.3 db)` 

### 2. **Accuracy measurement for testset.**

Console output using provided small test set may looks like: 

       ACCURACY CALCULATION on Input IDX testset according to IDX labels set
       IDX test file shape: [30,128,9,]
       Model input shape: [128,9,]

              3 of 30 test vectors are processed (2 are correct: 66.667 %)
              6 of 30 test vectors are processed (5 are correct: 83.333 %)
              9 of 30 test vectors are processed (8 are correct: 88.889 %)
              12 of 30 test vectors are processed (10 are correct: 83.333 %)
              15 of 30 test vectors are processed (12 are correct: 80.000 %)
              18 of 30 test vectors are processed (15 are correct: 83.333 %)
              21 of 30 test vectors are processed (18 are correct: 85.714 %)
              24 of 30 test vectors are processed (21 are correct: 87.500 %)
              27 of 30 test vectors are processed (24 are correct: 88.889 %)
              30 of 30 test vectors are processed (27 are correct: 90.000 %)
       Final Accuracy: 90.000 % (27 are correct of 30)
       FINISHED

where:
* `Final Accuracy: 90.000 % (27 are correct of 30)` reflects how much samples from the testset was accurately predicted in comparison with reference label. The accuracy itself for the provided small test dataset should be  `90.000 %` for any  `run_*` command.

### 3 **External test-set processing.**
Console output using provided small test set should looks mostly the same as for accuracy measurements mode, but without accuracy values.

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

# Data Memory Requirements

Example application uses statically allocated memory for model weights and intermediate results (activations) and structures. Requirements for them depends on model bit depth 
configuration define and listed in table below. Before compiling application for desired hardware configuration, be sure it has enough memory to store the data.

|                      Data                         |   MODEL_BIT_DEPTH=8   |  MODEL_BIT_DEPTH=816  |  MODEL_BIT_DEPTH=16  |
| :-----------------------------------------------: | :-------------------: | :-------------------: | :------------------: |
| Weights <br/>(*.mli_model* section)              |  18040 bytes          | 17160 bytes           | 34316 bytes          |
| Activations <br/>(*.Xdata* and *.Ydata* sections) |  14880 bytes           | 18752 bytes           | 18752 bytes          |

By default, application uses MODEL_BIT_DEPTH=16 mode. Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

# References

GitHub project served as starting point for this example:
> Guillaume Chevalier, *LSTMs for Human Activity Recognition*, 2016,[https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)

Human Activity Recognition Using Smartphones [Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones):
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *"A Public Domain Dataset for Human Activity Recognition Using Smartphones."* 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013:

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
