CIFAR-10 Convolution Neural Network Example 
==============================================
Example is based on standard [Caffe tutorial](http://caffe.berkeleyvision.org/gathered/examples/cifar10.html) for [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. It's a simple classifier built on convolution, pooling and dense layers for tiny images.

# Building and Running

You need to configure and build the library project for the desired platform. 
Please read the corresponding section on [building the package](/README.md#building-the-package). 
There are no extra requirements specific for this application. All the specified platforms are supported by the test application.  

Build artifacts of the application are stored in the `/obj/<project>/examples/example_cifar10_caffe` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps. 
You need to replace `<options>` placeholder in commands below with the same options list you used for the library configuration and build. 

1. Open command line in the root of the embARC MLI repo and change working directory to './examples/example_cifar10_caffe/'

       cd ./examples/example_cifar10_caffe/

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build the example. This is an optional step as you may go to the next step which automatically invokes the build process. 

       gmake <options> build

4. Run the example

       gmake <options> run

## More Options on Running

Coefficients for trained NN model are stored in a separate compile unit as wrapped float numbers or integer quantized numbers in case of SA8. This allows coefficients to be transformed into quantized fixed point values at compile time.
For this reason you can build and check application with 8 and 16 bit depth of NN coefficients and data. General run template is the following:

       gmake <options> <run_target>

where `<options>` is defined earlier in this file and `run_target` might be one of the following:

* `run` : same as `run_FX16`
* `run_FX16` :  16 bit depth of coefficients and data (FX16) (default):
* `run_SA8` : 8 bit depth of coefficients and data (SA8):
* `run_FX16_FX8_FX8` : 8x16: 8 bit depth of coefficients and 16 bit depth of data (FX8 weights and FX16 data):


Example application may be used in three modes:
1. **Built-in input processing.** Uses only hard-coded vector for the single input model inference. 
No application input arguments.

       gmake <options> <run_target>

2. **External test-set processing.** Reads vectors from input IDX file, passes it to the model, and writes its output to the other IDX file (if input is *tests.idx* then output will be *tests.idx_out*). 
Input test-set path is required as argument

       gmake <options> <run_target> RUN_ARGS="small_test_base/tests.idx"

3. **Accuracy measurement for testset.** Reads vectors from input IDX file, passes it to the model, and accumulates number of successive classifications according to labels IDX file. 
Input test-set and labels paths are required as argument.

       gmake <options> <run_target> RUN_ARGS="small_test_base/tests.idx small_test_base/labels.idx"


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
    cd ./examples/example_cifar10_caffe
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
    cd ./examples/example_cifar10_caffe
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
       ir_conv1.idx(w/o IR check):     X cycles
       ir_pool1.idx(w/o IR check):     X cycles
       ir_conv2.idx(w/o IR check):     X cycles
       ir_pool2.idx(w/o IR check):     X cycles
       ir_conv3.idx(w/o IR check):     X cycles
       ir_pool3.idx(w/o IR check):     X cycles
       ir_ip1.idx(w/o IR check):       X cycles
       ir_prob.idx(w/o IR check):      X cycles


       Summary:
              Layer1: X cycles
              Layer2: X cycles
              Layer3: X cycles
              Layer4: X cycles
              Layer5: X cycles

              Total: X cycles

       Result Quality: S/N=4765.1     (73.6 db)
       FINISHED

where:
* `X cycles` reflects number of cycles for a specific layer or in total . `X` may vary depending on target platform and build options.

* `Result Quality: S/N=4765.1      (73.6 db)` reflects the signal-to-noise ration of the model output in comparison with reference float. The ratio itself (`S/N` and `x db`) may vary depending on the target platform and `run_*` command. In particular :

  * `run_FX16`: Result may slightly fluctuate around `S/N=4765.1     (73.6 db)` 
  * `run_FX16_FX8_FX8`: Result may slightly fluctuate around `S/N=33.6       (30.5 db)` 
  * `run_SA8`: Result may slightly fluctuate around `S/N=61.9       (35.8 db)` 

### 2. **Accuracy measurement for testset.**

Console output using provided small test set may look like: 

       ACCURACY CALCULATION on Input IDX testset according to IDX labels set
       IDX test file shape: [20,32,32,3,]
       Model input shape: [32,32,3,]

              2 of 20 test vectors are processed (2 are correct: 100.000 %)
              4 of 20 test vectors are processed (4 are correct: 100.000 %)
              6 of 20 test vectors are processed (6 are correct: 100.000 %)
              8 of 20 test vectors are processed (8 are correct: 100.000 %)
              10 of 20 test vectors are processed (10 are correct: 100.000 %)
              12 of 20 test vectors are processed (12 are correct: 100.000 %)
              14 of 20 test vectors are processed (14 are correct: 100.000 %)
              16 of 20 test vectors are processed (16 are correct: 100.000 %)
              18 of 20 test vectors are processed (18 are correct: 100.000 %)
              20 of 20 test vectors are processed (20 are correct: 100.000 %)
       Final Accuracy: 100.000 % (20 are correct of 20)
       FINISHED

where:
* `Final Accuracy: 100.000 % (20 are correct of 20)` reflects how many samples from the testset were accurately predicted in comparison with reference label. The accuracy itself may vary depending on the target platform and `run_*` command. In particular :

  * `run_FX16` and `run_FX16_FX8_FX8`: Accuracy should be  `100.000 %` for provided small test dataset. 
  * `run_SA8`:  Accuracy should be  `90.000 %` for provided small test dataset.

### 3 **External test-set processing.**
Console output using provided small test set should looks mostly the same as for accuracy measurement mode, but without accuracy values.

# Example Structure

Structure of example application may be logically divided into three parts:

* **Application.** Implements Input/output data flow and data processing by the other modules. Application includes
   * ml_api_cifar10_caffe_main.c
   * ../auxiliary/examples_aux.h(.c)
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model-related constants are pre-defined and model coefficients are declared in the separate compile unit 
   * cifar10_model.h
   * cifar10_model_hwcn.c
   * cifar10_constants.h
   * cifar10_coefficients_hwcn.c
* **Auxiliary code.** Various helper functions for measurements, IDX file IO, etc.
   * ../auxiliary/tensor_transform.h(.c)
   * ../auxiliary/tests_aux.h(.c)
   * ../auxiliary/idx_file.h(.c)

Example structure contains test set including small subset of CIFAR-10 (20 vectors organized in IDX file format).

# Data Memory Requirements


Example application uses statically allocated memory for model weights and intermediate results (activations) and structures. Requirements for them depends on model bit depth 
configuration define and listed in table below. Before compiling application for desired hardware configuration, be sure it has enough memory to meet the data requirements.

|                      Data                              |   MODEL_BIT_DEPTH=8   |  MODEL_BIT_DEPTH=816  |  MODEL_BIT_DEPTH=16  |
| :----------------------------------------------------: | :-------------------: | :-------------------: | :------------------: |
| Weights <br/>*.mli_model* and *mli_model_p2* sections  |  33212 bytes          | 33212 bytes           | 66420 bytes          |
| Activations 1 <br/>*.Zdata* section                    |  32768 bytes          | 65536 bytes           | 65536 bytes          |
| Activations 2 <br/>*.Ydata* section                    |  8192 bytes           | 16384 bytes           | 16384 bytes          |
| Activations 3 <br/>*.Xdata* section                    |  1024 bytes           | 1024  bytes           | 1024  bytes          |

By default, application uses MODEL_BIT_DEPTH=16 mode. Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

# References

CIFAR-10 Dataset:
> Alex Krizhevsky. *"Learning Multiple Layers of Features from Tiny Images."* 2009.

Caffe framework:
> Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor. *"Caffe: Convolu-tional Architecture for Fast Feature Embedding."* arXiv preprint arXiv:1408.5093. 2014: http://caffe.berkeleyvision.org/

IDX file format originally was used for [MNIST database](http://yann.lecun.com/exdb/mnist/). There is a python [package](https://pypi.org/project/idx2numpy/) for working with it through transformation to/from numpy array. *auxiliary/idx_file.c(.h)* is used by the test app for working with IDX files:
> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. *"Gradient-based learning applied to document recognition."* Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
