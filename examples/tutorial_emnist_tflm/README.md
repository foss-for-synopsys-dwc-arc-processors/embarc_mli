# EMNIST Example: Tensorflow Lite Micro Usage

This example shows how to use Tensorflow Lite Micro together with an embARC MLI application. It consists of two parts:

1) Example application. It demonstrates how to use the tflite-micro API and guides you through the building aspects of the application and libraries.

2) Conversion Tutorial. An independent part showing how the NN model for the application was converted into tflite format and adapted to be MLI compatible.

The first part is disclosed in this readme. The details of the conversion tutorial are covered in the separate readme file in the [*conversion_tutorial*](/examples/tutorial_emnist_tflm/conversion_tutorial) directory. Passing the second part is not necessary to run the example application.

 **Important notes:**

* Example is supported only for VPX configurations with guard bits. For EM/HS, please use [MLI 1.1 release version](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_1.1/examples/tutorial_emnist_tflm).

## Generate Tensorflow Lite Micro Library

To build and run the example application at first you need to generate **tflite-micro** static library.

Tensorflow Lite for Microcontrollers is a separate project with specific set of requirements. We recommend to use [embARC fork](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro) of Tensorflow Lite for Microcontrollers repository:

    git clone https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro.git

The fork has been updating periodically from the [upstream repo](https://github.com/tensorflow/tflite-micro) using states that are stable in relation to ARC target.

Please first familiarize yourself with [TFLM ARC specific details](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/targets/arc/README.md) and make sure that your environment is set up appropriately.

Important information is listed inside [make tool section](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro/tree/main/tensorflow/lite/micro/tools/make/targets/arc#make-tool) of the referred document.
The main message is that native *nix environment is required to build the TFLM library.
For Windows users there are no officially supported flow.
You still may consider projects like [WSL](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) at your own risk.

To build **tflite-micro** library please find the corresponding section in documentation specified for a [custom ARC platform](https://github.com/foss-for-synopsys-dwc-arc-processors/tflite-micro/tree/main/tensorflow/lite/micro/tools/make/targets/arc#Custom-ARC-EMHSVPX-Platform).  You need to copy the generated library to the *third_party* directory of this example and rename it to *libtensorflow-microlite.a* (see the same documentation on where the generated library can be found).

For the following example application build you should set ``TENSORFLOW_DIR``
environment variable to point to the cloned tflite-micro repository:

    # For Windows
    set TENSORFLOW_DIR=<your-path-to-tflite-micro>

    # For Linux
    export TENSORFLOW_DIR=<your-path-to-tflite-micro>

## Building and Running Example

After you've passed ["Generate Tensorflow Lite Micro Library"](#generate-tensorflow-lite-micro-library) step
you need to configure and build the library project for the desired VPX
configuration. Please read the corresponding section on [building the package](/README.md#building-the-package). **Also make sure you didn't forget to set ``TENSORFLOW_DIR`` environment variable.**

Build artifacts of the application are stored in the `/obj/<project>/examples/tutorial_emnist_tflm` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps.
You need to replace `<options>` placeholder in commands below with the same [build configuration options](README.md#build-configuration-options) list you used for the library configuration and build.

1. Open command line in the root of the embARC MLI repo and change working directory to './examples/tutorial_emnist_tflm/'

       cd ./examples/tutorial_emnist_tflm/

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build the example. This is an optional step as you may go to the next step which automatically invokes the build process.

       gmake <options> build

4. Run the example

       gmake <options> run

## ARC VPX Build Process Example

Assuming you've already built and copied *libtensorflow-microlite.a* and your environment satisfies all build requirements for ARC VPX platform, you can use the following script to build and run the application using the nSIM simulator.
The first step is to open a command line and change working directory to the root of the embARC MLI repo.

1. Clean all previous artifacts for all platforms

       gmake cleanall

1. Generate recommended  TCF file for VPX

       tcfgen -o ./hw/vpx5_integer_full.tcf -tcf=vpx5_integer_full -iccm_size=0x80000 -dccm_size=0x40000

1. Build project using generated TCF and appropriate built-in runtime library for it. Use multithreaded build process (4 threads):

       gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full JOBS=4 build

1. Change working directory and build the example:

       cd ./examples/tutorial_emnist_tflm
       gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full build

1. Run the example:

       gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run
