Face Detect Example
==============================================

This example is based on the [BlazeFace](#references): fast and light-weight face detector.
It shows how advanced but still compact models might be implemented utilizing limited CCM memory efficiently.

Slicing logic helps to split calculation of large intermediate feature maps into parts 
propagating it through the network architecture towards layers with more compact output. 

# Building and Running

You need to configure and build the library project for the desired platform. 
Please read the corresponding section on [building the package](/README.md#building-the-package). 
There are no extra requirements specific for this application. All the specified platforms are supported by the test application.  

Build artifacts of the application are stored in the `/obj/<project>/examples/example_face_detect` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps. 
You need to replace `<options>` placeholder in commands below with the same options list you used for the library configuration and build. 

1. Open command line in the root of the embARC MLI repo and change working directory to './examples/example_face_detect/'

       cd ./examples/example_face_detect/

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build the example. This is an optional step as you may go to the next step which automatically invokes the build process. 

       gmake <options> build

4. Run the example

       gmake <options> run RUN_ARGS=<bmp_image_path>

where ``<bmp_image_path>`` is the path to a supported 128x128 RGB file in BMP format. You can start with the provided ``test_img.bmp`` file.

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
    cd ./examples/example_face_detect
    gmake ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 build
    ```

4. Run example using ``test_img.bmp`` as input:
    ```bash
    gmake ROUND_MODE=UP FULL_ACCU=OFF run RUN_ARGS=test_img.bmp
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
    cd ./examples/example_face_detect
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full build
    ```

5. Run example using ``test_img.bmp`` as input:
    ```bash
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full RUN_ARGS=test_img.bmp run
    ```

## Expected Output

Application will create `result.bmp` file in the working directory. It's the same input file with framed faces which was found in run. Expected console output for a ``test_img.bmp`` is the following: 

     Pre_process ticks: <X>
     Model ticks: <X>
     Post_process ticks: <X>
     Total ticks: <X>
     Found a face at ([X:11, Y:44]; [X:30, Y:63]) with (0.997989) score
     Found a face at ([X:99, Y:42]; [X:118, Y:60]) with (0.996663) score
     Found a face at ([X:53, Y:44]; [X:70, Y:61]) with (0.987044) score
     Done. See result in "result.bmp" file.

where `<X>` reflects number of cycles for a specific layer or in total . `X` may vary depending on target platform and build options.

# Example Structure
Structure of example application may be divided logically on three parts:

* **Application.** Implements resources allocation, reading and writing BMP files and invoking face detect model with all related pre/post processing:
   * bmp_file_io.c
   * bmp_file_io.h
   * face_detect_module.cc
   * face_detect_module.h
   * main.c
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model related constants are pre-defined and model coefficients is declared in the separate compile unit 
   * model/*.cc
   * model/*.h
   
Example structure contains test image of Geoffrey Hinton, Yoshua Bengio and Yann Lecun (see *test_img.bmp* in [references below](#references)). The image was downscaled to 128x128 pixels preserving aspect ratio.

# Data Memory Requirements

Example uses statically allocated memory for model weights and intermediate results (activations). Before compiling application for desired hardware configuration, be sure it has enough memory to store the data. For images on the application level (outside of the model) 
example allocates memory dynamically before processing.

|                      Data                         |         Size          |
| :-----------------------------------------------: | :-------------------: |
| Weights in CCM memory (constant)                  |  ~62 Kbytes           |
| Weights in system memory (constant)               |  ~56 Kbytes           |
| Anchors in system memory (constant)               |  ~14 Kbytes           |
| Activations CCM memory (BSS)                      |  ~175 Kbytes          |
| Pre/Post processing data in system memory (BSS)   |  ~129 Kbytes          |

Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

# References

Original paper for the BlazeFace detector:
>Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran and Matthias Grundmann. *BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs*. arXiv preprint [arXiv:1907.05047v2](https://arxiv.org/abs/1907.05047), 2019

GitHub project served as starting point for this example:
> Matthijs Hollemans, *BlazeFace-PyTorch: The BlazeFace face detector model implemented in PyTorch*, 2019, [link](https://github.com/hollance/BlazeFace-PyTorch)

*test_img.bmp*:
> Geoffrey Hinton, Yoshua Bengio, Yann Lecun. In [AIBuilders](https://aibuilders.ai/le-prix-turing-recompense-trois-pionniers-de-lintelligence-artificielle-yann-lecun-yoshua-bengio-et-geoffrey-hinton/), 2019,



