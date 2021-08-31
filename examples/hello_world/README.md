Hello World Example
==============================================
This is an introductory example showing basic usage of MLI data structures and kernels.

# Building and Running

You need to configure and build the library project for the desired platform. 
Please read the corresponding section on [building the package](/README.md#building-the-package). 
There are no extra requirements specific for this application. All the specified platforms are supported by the test application.  

Build artifacts of the application are stored in the `/obj/<project>/examples/hello_world` directory where `<project>` is defined according to your target platform.  

After you've built and configured the whole library project, you can proceed with the following steps. 
You need to replace `<options>` placeholder in commands below with the same options list you used for the library configuration and build. 

1. Open command line in the root of the embARC MLI repo and change working directory to './examples/hello_world/make/'

       cd ./examples/hello_world/make

2. Clean previous build artifacts (optional).

       gmake <options> clean

3. Build the example. This is an optional step as you may go to the next step which automatically invokes the build process. 

       gmake <options> build

4. Run the example

       gmake <options> run

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
    cd ./examples/hello_world/make
    gmake ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 build
    ```

4. Run example:
    ```bash
    gmake ROUND_MODE=UP FULL_ACCU=OFF run 
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
    cd ./examples/hello_world/make
    gmake TCF_FILE=../../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full build
    ```

5. Run example:
    ```bash
    gmake TCF_FILE=../../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full run
    ```

## Expected Output

The same console output is expected for any build configuration: 

    in1:
    1 2 3 4 5 6 7 8
    in2:
    10 20 30 40 50 60 70 80
    mli_krn_eltwise_add_fx16 output:
    11 22 33 44 55 66 77 88
    mli_krn_eltwise_sub_fx16 output:
    -9 -18 -27 -36 -45 -54 -63 -72

# Example Structure

All the application code is concentrated inside the only ``main.cpp`` file. There are no other application level dependencies outside this directory beside MLI Library itself.


# Data Memory Requirements

Application is configured to use 8 KBytes of heap and stack in total. 
Application data which must be held in fast memory (kernel operands) is less than 100 bytes. Code size depends on the platform the application is built for.


