embARC Machine Learning Inference Library
==================================================

This repository contains source code of embARC Machine Learning Inference Library (embARC MLI Library), its documentation and examples. The primary purpose of this library is to enable developers to efficiently implement and/or port data processing algorithms based on machine learning principles for DSP-enhanced ARC Processors.

# Table of Content

- [Release Notes](#release-notes)
- [Documentation](#documentation)
- [Package Structure](#package-structure)
- [Quick Start](#quick-start)
- [Building the Package](#building-the-package)
  - [General Build Process](#general-build-process)
  - [x86 Host Emulation](#x86-host-emulation)
  - [ARC Processors](#arc-processors)
  - [Build Configuration Options](#build-configuration-options)
- [Examples And Tests](#examples-and-tests)
- [Known Issues](#known-issues)
- [Frequently Asked Questions](#frequently-asked-questions)


# Release Notes

1. Version 2.0

3. This release supports following functional primitives
    * 2D Convolution
    * 2D Depthwise Convolution
    * 2D Transpose Convolution
    * 2D Group Convolution
    * Fully Connected layer
    * Max and average pooling
    * LSTM and GRU recurrent cells 
    * RNN Dense layer
    * Elementwise (add, sub, mul, min, max)
    * Permute
    * Argmax
    * Data manipulation (concatenation, permute, 2D padding)
    * ReLU, Leaky ReLU, Parametric ReLU, ReLU1, ReLU6
    * Softmax, Sigmoid, TanH, L2 Normalization
    * Helper functions to copy (partial) tensors (mli_mov*)

3. Supported data layout:
    * Data layout HWC (Height-Width-Channel)

4. Supported data format:
    * Fixed point 8bit and 16bit (fx8 and fx16)
    * Signed asymmetric 8bit quantization (sa8)
    * Signed asymmetric datatype supports per-tensor or per channel quantization with 16bit scale factors.

5. Slicing support: creation of sub-tenors and support for non-contiguous tensor data.

6. Supported platforms:
    * VPX
    * x86 emulation

6. Toolchains support:
    * [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) version 2021.03 and newer.
    * GCC 9.1.0 (for x86 emulation)
    * MSVC 2019 (for x86 emulation)

# Documentation

embARC MLI library API documentation for version 2.0 is [available online](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_mli/doc/build/html/index.html) starting from the release date. 

It's sources are available in the [/doc](/doc) directory and can be built as described in the related [readme file](doc/README.md). 



# Package Structure

`./bin`                               	- directory for embARC MLI library binary archive created during build  
`./obj`                               	- directory for all intermediate artifacts created during build  
`./cmake`                             	- contains CMake settings file  
`./make`                              	- contains common GNU make rules and settings  
`./doc`                             		- contains the API documentation of the embARC MLI library  
`./include`                         		- include files with API prototypes and types  
`./lib/src`                         		- source code of embARC MLI Library  
`./lib/gen`                         		- auxiliary generation scripts for LUT tables  
`./lib/make`                        		- makefiles for library and package  
`./user_tests`                      		- set of basic tests for library primitives  
`./examples`                        		- source code of examples (see [Examples and Tests](#examples-and-tests)) section  
`./examples/auxiliary`               		- source code of helper functions used for the examples  
`./hw`                              		- contains HW templates (*.tcf files). See related [readme file](/hw/README.md)   

# Quick Start

Quick start guide is not yet defined. If you don't want to read the whole readme, as a compromise you can proceed with the following steps, to first build the MLI library for x86 emulation:

1. Make sure your environment satisfies the requirements from the [General Build Process](#general-build-process) and [x86 Host Emulation](#x86-host-emulation) sections.
2. Go to the [Build Command Examples For x86](#build-command-examples-for-x86) section, read it, and choose one of the proposed options for build. 
3. Go to the [Examples And Tests](#examples-and-tests) section, read it, and choose one of the listed examples for the next steps on running something with embARC MLI Library.

As the next step, you can repeat this recipe for ARC processors:

1. Make sure your environment satisfies the requirements from the [General Build Process](#general-build-process) and [ARC Processors](#arc-processors) sections.
2. Go to the [Build Command Examples For ARC Processors](#build-command-examples-for-arc-processors) section, read it, and choose one of the proposed options for build. 
3. Go to the [Examples And Tests](#examples-and-tests) section, read it, and choose one of the listed examples for the next steps on running something with embARC MLI Library.

Afterward you can continue with familiarizing yourself with [the documentation](#table-of-content), which contains all the necessary info and references. 

**Note that it is highly recommended to use DBG_MODE_DEBUG configuration option (see [`MLI_DEBUG_MODE`](#mli_debug_mode)) for early development of applications based on embARC MLI Library because it provides additional diagnostic output which can help you quickly track down misuse of the API**.

# Building the Package

The embARC MLI Library uses [CMake](https://cmake.org/) as a backend for the platform independent project generation and [GNU Make](https://www.gnu.org/software/make/) as a front end to invoke CMake and to run tests. Alternatively, after CMake configures the project for the desired platform, you can work with its output stored in `obj` folder as you may be used to. 

## General Build Process

Basic build requirements are the following:
 - [CMake](https://cmake.org/) version 3.18 or higher
 - [GNU Make](https://www.gnu.org/software/make/) version 3.82 or higher. 

A compatible version of `gmake` is also delivered with the [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) 2020.12 and higher. All command examples in the repo readmes will use `gmake`, but you can replace it with your suitable and compatible one.

The embARC MLI front end Make infrastructure provides targets for easy configuration of the project for a desired platform and toolchain. It also can build from sources, run tests and examples.

General template of the build command looks like:

```bash
gmake <target> <options> 
```

Available `<targets>`:
- `build` - configure project and build binaries. 
- `clean` - delete binaries and object files for configured project.
- `cleanall` - delete all configured projects and binaries.

**Note that tests and example applications have separate makefiles with additional targets which aren't explained here. See [Examples and Tests](#examples-and-tests) section for info and references**

`<options>` are described in the [Build Configuration Options](#build-configuration-options) section below. Here is a list of links for available options:
 - [`ROUND_MODE`](#round_mode)
 - [`TCF_FILE`](#tcf_file)
 - [`BUILDLIB_DIR`](#buildlib_dir)
 - [`MLI_BUILD_REFERENCE`](#mli_build_reference)
 - [`FULL_ACCU`](#full_accu)
 - [`JOBS`](#jobs)
 - [`VERBOSE`](#verbose)
 - [`OPTMODE`](#optmode)
 - [`MLI_DEBUG_MODE`](#mli_debug_mode)
 - [`DEBUG_BUILD`](#debug_build)
 - [`RECONFIGURE`](#reconfigure)
 - [`GEN_EXAMPLES`](#gen_examples)

**Note that tests and example applications are part of the same generated CMake build system as the library itself which behavior depends on [build configuration options](#build-configuration-options)). Tests and examples also may be adjusted using their own configuration options not listed above. See [Examples and Tests](#examples-and-tests) section for info and references.**

The embARC MLI Library can be built for the following platforms:
- [x86 Host Emulation](#x86-host-emulation)
- [ARC Processors](#arc-processors)

Build for these platforms creates separate projects in `obj` directory and separate binaries in `bin` directory. Build process for supported platforms is defined below.


## x86 Host Emulation

The embARC MLI Library can be built for host platform and used in compatible applications to ease early development or verification. x86 and x64 architectures are supported, but for simplicity only x86 will be mentioned within documentation. No optimization is applied for this platform. Depending on the MLI build configuration, calculation results on x86 platform can be bit exact with desired ARC processor within defined behavior of MLI Functions.

The x86 Host emulation of the library has been tested with the following toolchains:
- GCC 9.1.0
- MSVC 2019

To build embARC MLI library	you need
1. Open command line and change working directory to the root of the repo
2. Start building using the following command template which you need to adjust for your needs. 

```bash
gmake build ROUND_MODE=[UP|CONVERGENT] <Additional options>
```

[`ROUND_MODE`](#round_mode) is a mandatory option for x86 host emulation target. [`TCF_FILE`](#tcf_file) option must not be used (only empty value is allowed).

As a result of configuration and build you will find `bin/native` folder with the library binary file and `obj/native` directory with generated project for the default toolchain and IDE within the environment.

`<Additional options>` which are applicable for this mode are [`JOBS`](#jobs), [`VERBOSE`](#verbose), [`FULL_ACCU`](#full_accu), [`MLI_DEBUG_MODE`](#mli_debug_mode), [`RECONFIGURE`](#reconfigure), [`GEN_EXAMPLES`](#gen_examples).

`<Additional options>` which have no effect or do not make sense in this mode are [`BUILDLIB_DIR`](#buildlib_dir), [`MLI_BUILD_REFERENCE`](#mli_build_reference), [`OPTMODE`](#optmode), [`DEBUG_BUILD`](#debug_build).


### **Build Command Examples for x86**

The first step is to open a command line and change working directory to the root of the embARC MLI repo. Afterward, you can use one of the following commands.

1. Build project to emulate ARC VPX platform:
    ```bash
    gmake build ROUND_MODE=UP FULL_ACCU=OFF 
    ```

2. Build project to emulate ARC VPX platform with full debug checking of parameters and assertions in runtime. Use multithreaded build process (4 threads):
    ```bash
    gmake build ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 MLI_DEBUG_MODE=DBG_MODE_FULL 
    ```


## ARC Processors

Main target platforms for embARC MLI Library are ARC processors. The specific processor family is determined by *.tcf file provided for library configuration. It is highly recommended to use embARC MLI 2.0 for VPX processor only. EM/HS targets are not properly tested and optimized. You can use [embARC MLI 1.1](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases/tag/Release_1.1) instead. 

embARC MLI Library build for ARC processors requires [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) (MWDT) version 2021.03 and higher.

To build embARC MLI library	you need
1. Open command line and change working directory to the root of the repo
2. Start building using the following command template which you need to adjust for your needs. 

```bash
gmake build TCF_FILE=<path_to_tcf> [BUILDLIB_DIR=<path_to_target_rt_libs>] <Additional options>
```

[`TCF_FILE`](#tcf_file) is a mandatory option for ARC target. 

**In case you are going to compile and run tests or examples it is better to provide the path to a runtime library using the [`BUILDLIB_DIR`](#buildlib_dir) option.**

As a result of configuration and build you will find `bin/arc` folder with the MLI library and `obj/arc` directory with generated Makefile project configured to use MWDT toolchain. If you use [`MLI_BUILD_REFERENCE`](#mli_build_reference) option, then artifacts will be created in `bin/arc_ref` and `obj/arc_ref` directories correspondingly. 

`<Additional options>` which are applicable for this mode are 
[`JOBS`](#jobs), [`VERBOSE`](#verbose), 
[`MLI_BUILD_REFERENCE`](#mli_build_reference), [`MLI_DEBUG_MODE`](#mli_debug_mode), 
[`DEBUG_BUILD`](#debug_build), [`RECONFIGURE`](#reconfigure), [`GEN_EXAMPLES`](#gen_examples), 
[`OPTMODE`](#optmode).

`<Additional options>` which have no or limited effect in this mode are [`FULL_ACCU`](#full_accu), [`ROUND_MODE`](#round_mode). [`ROUND_MODE`](#round_mode) option is applicable only for ARC EMxD family.


### **Build Command Examples for ARC Processors**
The following commands assume usage of the recommended VPX configuration. TCF for this configuration you need to generate using _tcfgen_ tool delivered with MetaWare Development tools, in order to ensure sufficient target memory to run all of the examples. The first step is to open a command line and change working directory to the root of the embARC MLI repo. Then use the following command to generate recommended tcf file taking default `vpx5_integar_full` configuration as basis:

```bash
tcfgen -o ./hw/vpx5_integer_full.tcf -tcf=vpx5_integer_full -iccm_size=0x80000 -dccm_size=0x40000
```

Afterward, you can use one of the following commands to configure and build the package:


1. Build project for recommended ARC VPX evaluation target. [`BUILDLIB_DIR`](#buildlib_dir) is mandatory for this, but default "vpx5_integer_full" pack delivered with MWDT tools can be used. Use multithreaded build process (4 threads):
    ```bash
    gmake TCF_FILE=./hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full JOBS=4 build 
    ```

2. Build project for recommended ARC VPX evaluation target optimized for code size and with full debug checking of parameters and assertions in runtime. Use multithreaded build process (4 jobs):
    ```bash
    gmake build TCF_FILE=./hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full \
	OPTMODE=size MLI_DEBUG_MODE=DBG_MODE_FULL JOBS=4
    ```

3. Build project for recommended ARC VPX evaluation target using reference code. It's unoptimized straightforward and expected to be bitwise with optimized one. Use multithreaded build process (4 jobs) and artifacts are stored in `bin/arc_ref` and `obj/arc_ref`:

    ```bash
    gmake build TCF_FILE=./hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full \
	MLI_BUILD_REFERENCE=ON JOBS=4
    ```


## Build Configuration Options

### `ROUND_MODE`
**Description**: Rounding mode for low level math on casting and shifting values.  

**Syntax**: `ROUND_MODE=[UP|CONVERGENT]`  
**Values**:  
 - `UP` - Rounding half up to the bigger value  
 - `CONVERGENT` - Rounding to the even value.  

**Default**:  
 - x86 Host Emulation: No default value. Mandatory option (to be set by user)
 and should generally be aligned to the ARC target you want to emulate in order to ensure consistent results.  
 - ARC EM / HS: `CONVERGENT` is a default. Might be changed.  
 - ARC VPX: `UP` is a default and the only option. Can't be changed.  


### `FULL_ACCU`
**Description**:  Usage of full or reduced accumulator bit depth during accumulation. This option is provided to emulate VPX specific low level optimization and can be used only for x86 platform, or together with [`MLI_BUILD_REFERENCE=ON`](#mli_build_reference) option.

**Syntax**: `FULL_ACCU=[ON|OFF]`  
**Values**:
 - `ON` - Use full accumulator bit depth.  
 - `OFF` - Use reduced accumulator bit depth.  
 
**Default**: `ON`  


### `TCF_FILE`
**Description**: Tool configuration file (TCF) file path. 

The TCF file defines ARC target processor hardware configuration. 
You can supply your own TCF that aligns with your hardware configuration. If you don't have a specific tcf file, you can use the _tcfgen_ util delivered with MetaWare Development tools. Basic _vpx5_integer_full_ template delivered with MetaWare Development tools might be a good starting point. The _tcfgen_ tool is documented in Linker and Utils guide which is also delivered with MetaWare Development tools. This option is mandatory for ARC platform and must not be set for x86 host emulation.  


**Syntax**: `TCF_FILE=<tcf-file>`  
**Values**: one of two options:
 - Path to a TCF file for a specific ARC target.
 - The name of the TCF file within the Metaware distribution.

**Default**: No default value.  


### `BUILDLIB_DIR`
**Description**: Path to runtime libraries for the ARC platform to link applications with.  

Runtime Libraries are required for [tests and example applications](#examples-and-tests) delivered with MLI Library package, but not needed for the library build. 
While for some targets not setting this option is acceptable (EMxD), it's highly recommended to build libraries specifically for your target. It can be done using the _buildlib_ util delivered with MetaWare Development tools. The _buildlib_ tool is documented in Linker and Utils guide which is also delivered with MetaWare Development tools. 
Alternatively, you can also pass the name of the runtime library delivered with MetaWare Development tools if it is compatible with your hardware configuration. For instance, together with `TCF_FILE=hw\vpx5_integer_full.tcf` you can use `BUILDLIB_DIR=vpx5_integer_full` to use compatible pre-built libraries within the Metaware distribution. 

This option has no effect on x86 host emulation build.  
  
**Syntax**: `BUILDLIB_DIR=<target_rt_libs>`  
**Values**: one of two options:
 - Path to a pre-built runtime libraries for a specific ARC target.
 - The name of the runtime library within the Metaware distribution.  

**Default**: No default value. The MetaWare compiler choses a default library for the ARC platform, which could result in incompatibilities with the TCF-file you specified. In case you are going to compile and run tests or examples for the ARC platform, it's better to provide a runtime library path.   


### `MLI_BUILD_REFERENCE`
**Description**: Switch embARC MLI Library implementation between platform independent reference code and platform default code.  

Reference code is a configurable straightforward unoptimized implementation. It's goal is to emulate desired ARC processor on the bit exact level within defined behavior of MLI Functions.
If this switch is turned on, artifacts will be generated into directory with `_ref` postfix (`bin/arc_ref` and `obj/arc_ref` for instance)  

**Syntax**: `MLI_BUILD_REFERENCE=[ON|OFF]`  
**Values**:  
 - `ON` - Use reference implementation of the library.  
 - `OFF` - Use default implementation of the library for the platform (optimized for ARC and reference for x86).  

**Default**: `OFF`  


### `JOBS`
**Description**: Number of jobs (threads) used on workstation to build the MLI package. Increasing number of jobs can reduce build time.  
**Syntax**: `JOBS=<number of jobs>`  
**Values**: It is recommended to use value within `[1; number of host logical cores]` range.  
**Default**: no default value (single thread)  


### `VERBOSE`
**Description**: Activates verbose output from CMake and build tools during build of the project.  

**Syntax**: `VERBOSE=[ON|OFF]`  
**Values**:  
 - `ON` - Activates verbose output.  
 - `OFF` - Disables verbose output.

**Default**: `OFF`  

### `OPTMODE`
**Description**: Define optimization mode of embARC MLI Library and all delivered examples and tests. This option has no effect on x86 host emulation build. 

**Syntax**: `OPTMODE=[speed|size]`  
**Values**:
 - `speed` - Build for optimal speed.  
 - `size` - Build for optimal code size.

**Default**: `speed`  

### `MLI_DEBUG_MODE`
**Description**: Additional debug features mode.

To ease application debugging, additional debug features can be turned-on during build which includes: 
 - _Return Codes_ - examine function parameters and return a valid status code.
 - _Messages_ - print a particular reason of stopping execution in `stdout` in case there is something wrong with function parameters. 
 - _Assertions_ - halt execution in case some data assumptions are not met including function parameters and internal invariants.
 
 For more info see *Debug Mode* section of *MLI API Data* chapter in the [MLI API Documentation](#documentation).

 Note that **it is highly recommended** to use `DBG_MODE_DEBUG` configuration option for early development of applications based on embARC MLI Library because it provides additional diagnostic output which can help you quickly track down misuse of the API.

**Syntax**: `MLI_DEBUG_MODE=DBG_MODE_[RELEASE|RET_CODES|ASSERT|DEBUG|FULL]`  
**Values**:
 - `DBG_MODE_RELEASE` - No debug. Messages:OFF; Assertions:OFF; ReturnCodes: Always OK.
 - `DBG_MODE_RET_CODES` - Return codes mode. Messages:OFF; Assertions:OFF; ReturnCodes: Valid Return.
 - `DBG_MODE_ASSERT` - Assert. Messages:OFF; Assertions:ON; Extra Assertions:OFF; ReturnCodes: Valid Return.
 - `DBG_MODE_DEBUG` - Debug. Messages:ON; Assertions:ON; Extra Assertions:OFF; ReturnCodes: Valid Return.
 - `DBG_MODE_FULL` - Full Debug. Messages:ON; Assertions:ON; Extra Assertions:ON; ReturnCodes: Valid Return.  

**Default**: `DBG_MODE_RELEASE`  

### `DEBUG_BUILD`
**Description**: Include debug information into binaries during build (`-g` flag). This option has no effect on x86 host emulation build.  

**Syntax**: `DEBUG_BUILD=[ON|OFF]`  
**Values**:  
 - `ON` - Include debug information into binaries.
 - `OFF` - Do not include debug information into binaries.

**Default**: `ON`  

### `RECONFIGURE`
**Description**: Always executes the CMake configure step, even if a project has already been configured. It may cause unwanted rebuilds.  

**Syntax**: `RECONFIGURE=[ON|OFF]`  
**Values**:
 - `ON` - Reconfigure project.
 - `OFF` - Do not reconfigure project.

**Default**: `OFF`  

 ### `GEN_EXAMPLES`
**Description**: Include MLI Examples (`./examples`) into the generation and build process together with the library and tests.   

**Syntax**: `GEN_EXAMPLES=[1|0]`  
**Values**:
 - `1` - Generate example projcets together with the library and tests.
 - `0` - Generate projects for the library and tests only.

**Default**: `1`  


# Examples And Tests

There are test and several examples supplied with embARC MLI Library. For information on how to build and run each example please go to the related directory and examine local README.

### [**User Tests**](/user_tests)
These are basic API level test applications to check that all the functions available at the API level work fine.

### [**Hello World**](/examples/hello_world)
This example is a first step API functions and data usage.

### [**CIFAR-10**](/examples/example_cifar10_caffe)
This example is a simple image classifier built on convolution, pooling and dense layers. It is based on standard Caffe tutorial for CIFAR-10 dataset.

### [**Human Activity Recognition**](/examples/example_har_smartphone)
LSTM Based Human Activity Recognition example. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone.

### [**Face Detection**](/examples/example_face_detect)
More advanced but still compact face detection example. It shows how the slicing and data movement can be organised to
efficiently use limited fast CCM memory.

### [**EMNIST TFLM Tutorial**](/examples/tutorial_emnist_tflm)
This example shows how to convert EMNIST Tensorflow model into Tensorflow Lite Micro format and use it in application.


<!-- 
## [Human Activity Recognition](/examples/example_har_smartphone)
LSTM Based Human Activity Recognition example. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone. 

## [Face Detection](/examples/example_face_detect)
Example shows basic implementation of the classic object detection (face detection in our case) via sliding window paradigm. 

## [Key Word Spotting](/examples/example_kws_speech)
An example of speech recognition implementation for key word spotting.
-->

# Known Issues

1. embARC MLI 2.0 is partially optimized for ARC EMxD and ARC HSxD targets. Currently we recommend only building for VPX and x86 emulation targets. You can use MLI 1.1 for EM/HS targets.


# Frequently Asked Questions

***Q: Can I use ARC GNU tools to build embARC MLI library?***  
A: No you cannot.<!-- embARC MLI Library must be built by MetaWare Development Tools only. Read the documentation at [github.io](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_mli/doc/build/html/getting_started/getting_started.html) for details-->

***Q: Can I use MetaWare Development Tools Lite to pre-build embARC MLI library and ARC GNU to build example application?***  
A: No you cannot. <!--embARC MLI Library must be built by full version of MetaWare Development Tools. Binaries built with MWDT Lite are not compatible with ARC GNU Tools and full MetaWare Development Tools. Read the MWDT Lite documentation for details.-->

***Q: I can not build and run example application for my Synopsys board (EMSK, IoTDK, etc), what I shall do?***  
A: It isn't supported at the moment. Currently we recommend only building for VPX and x86 emulation targets. You can use MLI 1.1 for EM/HS targets. <!--If you build for Synopsys boards refer to [platform documentation](https://foss-for-synopsys-dwc-arc-processors.github.io/platforms.html) as a good starting point. -->
<!--You should also note that example applications support different configurations for pre trained models and thus memory requirements, not all configurations can be built and run on Synopsys boards due to memory limitations and HW capabilities, read example application readme for details. embARC MLI Library must be also pre built specifically for your board by MetaWare Development Tools. Please note that makefiles provided with examples are configured for IoTDK only if GNU tools are used. -->

