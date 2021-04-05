embARC Machine Learning Inference Library
==================================================

:warning: **You are using a development branch. Things might be broken. For a proper usage of embARC MLI Library please checkout the [latest release](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_1.1).** 


This repository contains source code of embARC Machine Learning Inference Library (embARC MLI Library) documentation and examples. The primary purpose of this library is to enable developers to efficiently implement and/or port data processing algorithms based on machine learning principles for DSP-enhanced ARC Processors.

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

1. Early access of version 2.0
2. Release notes will be populated closer to the release date.

<!-- 3. This release supports following functional primitives
	* 2D Convolution
	* 2D depthwise convolution
	* Fully Connected layer
	* Max and average pooling
	* LSTM, Basic RNN
	* Elementwise (add, sub, mul, min, max)
	* Data manipulation (concatenation, permute, 2D padding)
	* ReLU, Leaky ReLu, ReLu1, ReLu6
	* Softmax, Sigmoid, TanH
	* Helper functions to copy (partial) tensors (mli_mov*)
4. Supported data layout:
	* CHW (Channel-Height-Width standard for Caffe)
	* Data layout HWC (Height-Width-Channel as used in TensorFlow Lite for Microcontrollers)
5. Supported data format:
	* Fixed point 8bit and 16bit (fx8 and fx16)
	* Signed asymmetric 8bit quantization (sa8) support for the following kernels:
		* Fully Connected
		* Convolution 2D (HWC Layout)
		* Depthwise Convolution 2D(HWC Layout)
		* Max Pooling(HWC Layout)
		* Average Pooling (HWC Layout)
6. Slicing support: creation of sub-tenors and support for non-contiguous tensor data. 
-->

# Documentation

embARC MLI library API documentation for version 2.0 is available in the [/doc](/doc) directory. It can be built from sources as described in the related [readme file](doc/README.md). 

The documentation will be available online closer to the release date. 
<!--Read the documentation at [github.io](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_mli/doc/build/html/index.html).-->

# Package Structure

<!-- Requires update on the examples side-->
`./bin`                               	- directory holder for embARC MLI library created during build  
`./obj`                               	- directory holder for the project created during configuration  
`./cmake`                             	- contains cmake settings file  
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

Quick start guide is not yet defined. If you don't want to read the whole readme, as a compromise you can proceed with the following steps:

1. Look on the requirements from the [General Build Process](#general-build-process) and [x86 Host Emulation](#x86-host-emulation) sections. Make sure your environment met them.
2. Go to the [Build Command Examples For x86](#build-command-examples-for-x86) section read it and choose one of the proposed options for build. 
3. Go to the [Examples And Tests](#examples-and-tests) section and chose one of the listed examples for the next steps on running something with embARC MLI Library.

As the next step, you can repeat this recipe for ARC Platform:

1. Look on the requirements from the [General Build Process](#general-build-process) and [ARC Processors](#arc-processors) sections. Make sure your environment met them.
2. Go to the [Build Command Examples For ARC Processors](#build-command-examples-for-arc-processors) section read it and choose one of the proposed option for build. 
3. Go to the [Examples And Tests](#examples-and-tests) section and chose one of the listed examples for the next steps on running something with embARC MLI Library.

Afterward you can continue with familiarizing yourself with [the documentation](#table-of-content) which contains all the necessary info and references. 

**Note that it is highly recommended to use DBG_MODE_DEBUG configuration option (see [`MLI_DEBUG_MODE`](#mli_debug_mode)) for early development of applications based on embARC MLI Library**.

# Building The Package

The embARC MLI Library uses CMake as a backend for the platform independent project generation and GNU Make as a frontend for configuring and invoking generation and build process. Alternatively, after CMake configure project for desired platform, you can work with it's output stored in `obj` folder as you used to. 

<!-- Currently the whole source tree represents the entire cmake project, so you need to provide all required options to generate the desired configuration of the project-->

<!-- Top level make files doesn't work, so it looks a little bit inconvinient to use lib make for the whole project configuration-->

## General Build Process

Basic build requirements are the following:
 - CMake version 3.18 or higher
 - GNU Make version 3.82 or higher. 

Compatible version of `gmake` is also delivered with the [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) 2020.12 and higher. All command examples in the repo readmes will use `gmake`, but you can replace it with your suitable and compatible one.

embARC MLI frontend Make infrastructure provides targets for ease configuration of the project for a desired platform and toolchain. It also can build sources, run tests and examples.

General template of build command looks like:

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

**Note that tests and example applications are also a part of generated projects and they also may be adjusted using their own configuration options. If their's default configuration isn't enough, these options must be provided together with other options for configuration. See [Examples and Tests](#examples-and-tests) section for info and references.**

embARC MLI Library can be built for the following platforms:
- [x86 Host Emulation](#x86-host-emulation)
- [ARC Processors](#arc-processors)

Build for these platforms creates separate projects in `obj` directory and separate binaries in `bin` directory . Build process for supported platforms is defined below.


## x86 Host Emulation

embARC MLI Library can be built for host x86 platform and used in compatible applications to ease early development or verification. Despite it is possible, no optimization is applied for this platform. Depending on MLI build configuration, calculation results on x86 platform can be bit exact with desired ARC processor within defined behavior of MLI Functions.

The x86 Host emulation of the library has been tested with the following toolchains:
- MSVC <!--(_2019 and later ==> versions?)-->
- GCC <!--( versions ?)-->
- LLVM <!--( versions ?)-->

<!-- Master Makefile must be fixed and related features should be refelcted here-->
Currently it's better to do configuration and build from the `lib/make` directory. Hence the first step is open command line and change working directory.

```bash
cd lib/make
gmake build ROUND_MODE=[UP|CONVERGENT] <Additional options>
```

[`ROUND_MODE`](#round_mode) is a mandatory option, while [`TCF_FILE`](#tcf_file) option must not be used (only empty value is allowed).

As a result of configuration and build you will find `bin/native` folder with the library binary file and `obj/native` directory  with generated project for the default toolchain and IDE within the environment.

`<Additional options>` with links to description which are applicable for this mode are [`JOBS`](#jobs), [`VERBOSE`](#verbose), [`FULL_ACCU`](#full_accu), [`MLI_DEBUG_MODE`](#mli_debug_mode), [`RECONFIGURE`](#reconfigure).

`<Additional options>` with links to description which may no effect or no sense in this mode are [`BUILDLIB_DIR`](#buildlib_dir), [`MLI_BUILD_REFERENCE`](#mli_build_reference), [`OPTMODE`](#optmode), [`DEBUG_BUILD`](#debug_build).


### **Build Command Examples For x86**

Just a reminder that currently it's better to do configuration and build from the `lib/make` directory. Hence the first step is open command line and change working directory.

```bash
cd lib/make
```

1. Project to emulate ARC EMxD platform:
    ```bash
    gmake build ROUND_MODE=CONVERGENT 
    ```

2. Project to emulate ARC VPX platform:
    ```bash
    gmake build ROUND_MODE=UP FULL_ACCU=OFF 
    ```

3. Project to emulate ARC VPX platform with full debug checking of parameters and assertions in runtime. Will be built in multi-thread mode (4 jobs):
    ```bash
    gmake build ROUND_MODE=UP FULL_ACCU=OFF JOBS=4 MLI_DEBUG_MODE=DBG_MODE_FULL 
    ```

4. Reconfigure and build existing x86 project to emulate ARC EM platform in 4 threads with debug checking (no asserts):
    ```bash
    gmake build ROUND_MODE=CONVERGENT \
	RECONFIGURE=ON JOBS=4 MLI_DEBUG_MODE=DBG_MODE_RET_CODES 
    ```


## ARC Processors

The main target platform for embARC MLI Library is ARC processors. The specific processor family is determined by *.tcf file provided for library configuration. Supported ARC Processor families:
- ARC VPX 
- ARC EMxD (currently unoptimized)
- ARC HSxD (currently unoptimized)

embARC MLI Library build for ARC processors requires [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) (MWDT) version 2021.03 and higher.

<!-- Master Readme must be fixed and related features should be refelcted here-->
Currently  it's better to do configuration and build from the `lib/make` directory. Hence the first step is open command line and change working directory.

```bash
cd lib/make
gmake build TCF_FILE=<path_to_tcf> [BUILDLIB_DIR=<path_to_target_rt_libs>] <Additional options>
```

[`TCF_FILE`](#tcf_file) is a mandatory option for ARC target. Package contains several TCF files in `/hw` directory  with recommended target configs for evaluation or for existing ARC based boards (see related [readme file](hw/README.md)). 

**Note In case you are going to compile and run tests or examples it's better to provide path to runtime libraries using [`BUILDLIB_DIR`](#buildlib_dir) option.**

As a result of configuration and build you will find `bin/arc` folder with the mli library and `obj/arc` directory  with generated Makefile project configured to use MWDT toolchain. If you use [`MLI_BUILD_REFERENCE`](#mli_build_reference) option, than artifacts will be created in `bin/arc_ref` and `obj/arc_ref` directories correspondingly. 

`<Additional options>` with links to description which are applicable for this mode are 
[`JOBS`](#jobs), [`VERBOSE`](#verbose), 
[`MLI_BUILD_REFERENCE`](#mli_build_reference), [`MLI_DEBUG_MODE`](#mli_debug_mode), 
[`DEBUG_BUILD`](#debug_build), [`RECONFIGURE`](#reconfigure), [`OPTMODE`](#optmode).

`<Additional options>` with links to description which may no or limited effect are [`FULL_ACCU`](#full_accu), [`ROUND_MODE`](#round_mode). [`ROUND_MODE`](#round_mode) option is applicable only for ARC EMxD family.


### **Build Command Examples For ARC Processors**

Just a reminder that currently it's better to do configuration and build from the `lib/make` directory. Hence the first step is open command line and change working directory.

```bash
cd lib/make
```

1. Project for recommended ARC VPX evaluation target. [`BUILDLIB_DIR`](#buildlib_dir) is mandatory for this, but default "vpx5_integer_full" pack delivered with MWDT tools can be used. Build in multithread mode (4 threads):
    ```bash
    gmake TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full JOBS=4 build 
    ```

2. Project for recommended ARC EM9D evaluation target. Default runtime libraries can be used. Build in multithread mode (4 threads):
    ```bash
    gmake TCF_FILE=../../hw/em9d.tcf JOBS=4 build
    ```

3. Project for recommended ARC VPX evaluation target optimized for code size and with full debug checking of parameters and assertions in runtime. Will be built in multi-thread mode (4 jobs):
    ```bash
    gmake build TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full \
	OPTMODE=size MLI_DEBUG_MODE=DBG_MODE_FULL JOBS=4
    ```

4. Reconfigure and build existing ARC project for recommended ARC EM9D evaluation target in 4 threads with debug checking (no asserts):
    ```bash
    gmake build TCF_FILE=../../hw/em9d.tcf  BUILDLIB_DIR=em9d_voice_audio \
	RECONFIGURE=ON MLI_DEBUG_MODE=DBG_MODE_RET_CODES JOBS=4 
    ```

5. Project for recommended ARC VPX evaluation target using reference code. It's  unoptimized straightforward and expected to be bitwise with optimized one. Will be built in multi-thread mode (4 jobs) and artifacts are stored in `bin/arc_ref` and `obj/arc_ref`:

    ```bash
    gmake build TCF_FILE=../../hw/vpx5_integer_full.tcf BUILDLIB_DIR=vpx5_integer_full \
	MLI_BUILD_REFERENCE=ON JOBS=4
    ```


## Build Configuration Options

### `ROUND_MODE`
**Description**: Rounding mode for low level math on casting and shifting values.  

**Syntax**: `ROUND_MODE=[UP|CONVERGENT]`  
**Values**:  
 - `UP` - Up rounding mode to the bigger value  
 - `CONVERGENT` - Rounding to the even value.  

**Default**:  
 - x86 Host Emulation: No default value. Mandatory option (to be set by user)  
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

The TCF file defines ARC target processor hardware configuration. [`./hw`](/hw) directory  contains several TCF files for evaluation. This option is mandatory for ARC platform and must not be set for x86 host emulation.  

**Syntax**: `TCF_FILE=<tcf-file>`  
**Values**: one of two options:
 - Path to a TCF file for a specific ARC target.
 - Name of the TCF file within The Metaware distribution.

**Default**: No default value.  


### `BUILDLIB_DIR`
**Description**: Path to runtime libraries used in application build for ARC platform.  

Runtime Libraries are required for [tests and example applications](#examples-and-tests) delivered with MLI Library package, but not needed for the library build. While for some targets not setting this option is acceptable (EMxD), it's highly recommended to build libraries specifically for your target. It can be done using buildlib util delivered with MetaWare Development tools. The buildlib tool is documented in Linker and Utils guide which is also delivered with MetaWare Development tools. This option has no effect on x86 host emulation build.  
  
**Syntax**: `BUILDLIB_DIR=<target_rt_libs>`  
**Values**: one of two options:
 - Path to pre-built runtime libraries for a specific ARC target.
 - Name of the runtime library within The Metaware distribution.  

**Default**: No default value which implies using default runtime libraries for ARC platform. Note that in case you are going to compile and run tests or examples for ARC platform it's better to provide path to runtime libraries using this option.   


### `MLI_BUILD_REFERENCE`
**Description**: Switch embARC MLI Library implementation between platform independent reference code and platform default code.  

Reference code is a configurable straightforward unoptimized implementation. It's goal is to emulate desired ARC processor on the bit exact level within defined behavior of MLI Functions.
If this switch is turned on, artifacts will be generated into directory  with `_ref` postfix (`bin/arc_ref` and `obj/arc_ref` for instance)  

**Syntax**: `MLI_BUILD_REFERENCE=[ON|OFF]`  
**Values**:  
 - `ON` - Use reference implementation of the library.  
 - `OFF` - Used default implementation of the library for the platform (optimized for ARC and reference for x86).  

**Default**: `OFF`  


### `JOBS`
**Description**: Number of jobs (threads) used for package build.  
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
 - _Return Codes_ - examine function parameters and return a valid code.
 - _Messages_ - print a particular reason of stopping execution in `stdout` in case something wrong with function parameters. 
 - _Assertions_ - halt execution in case some data assumptions are not met including function parameters and internal invariants.
 
 For more info see *Debug Mode* section of *MLI API Data* chapter in the [MLI API Documentation](#documentation).

 Note that **it is highly recommended** to use `DBG_MODE_DEBUG` configuration option for early development of applications based on embARC MLI Library.

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
**Description**: Force CMake to reconfigure already generated project which also leads to it's re-building. Doesn't affect other generated projects if they exist. For example, if you reconfiguring ARC specific project, already generated project for x86 host emulation won't be affected.  

**Syntax**: `RECONFIGURE=[ON|OFF]`  
**Values**:
 - `ON` - Reconfigure project.
 - `OFF` - Do not reconfigure project.

**Default**: `OFF`  


# Examples And Tests

There are test and several examples supplied with embARC MLI Library. For information on how to build and run each example please go to the related directory and examine local README.

### [**User Tests**](/user_tests)
This is a basic API level test applications to check that all the functions available at the API level work fine.

<!-- The embARC MLI Library is delivered with basic API level test application to check that all the functions available at the API level work fine.

(basic error measures checked with thresholds) and in the way that developers planned (CRC check of results).

There are several examples supplied with embARC MLI Library. For information on how to build and run each example please go to example directory and examine local README. -->

### [**CIFAR-10**](/examples/example_cifar10_caffe)
This example is a simple image classifier built on convolution, pooling and dense layers. It is based on standard Caffe tutorial for CIFAR-10 dataset.

<!-- 
## [Human Activity Recognition](/examples/example_har_smartphone)
LSTM Based Human Activity Recognition example. The model is intended to differentiate human activity between 6 classes based on inputs from embedded inertial sensors from waist-mounted smartphone. 

## [Face Detection](/examples/example_face_detect)
Example shows basic implementation of the classic object detection (face detection in our case) via sliding window paradigm. 

## [Key Word Spotting](/examples/example_kws_speech)
An example of speech recognition implementation for key word spotting.
-->

<!-- 
# Optimizations for code size

By default the embARC MLI Library is build for optimal speed. If code size needs to be reduced, there are two things that can be done:
1. For convolution and pooling layers there are specialized funtions for specific kernel sizes, they are called by a wrapper functions based on the parameters.
These parameters are compile time constant in the application, so the application can directly call the specialized functions. This will reduce over all code size.
Please be aware that the list of specializations is not guaranteed to be backwards compatible between releases.

2. Use a different optimization mode when calling the makefile. OPTMODE=size will optimize for size. default is OPTMODE=speed
	'gmake TCF_FILE=../../hw/em9d.tcf OPTMODE=size'
-->

# Known Issues

1. The embARC MLI 2.0 is in active development phase. Things might be broken, not optimal or contains bugs. For a proper usage of embARC MLI Library please checkout the [latest release](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_1.1).


# Frequently Asked Questions

***Q: Can I use ARC GNU tools to build embARC MLI library?***  
A: No you cannot.<!-- embARC MLI Library must be built by MetaWare Development Tools only. Read the documentation at [github.io](https://foss-for-synopsys-dwc-arc-processors.github.io/embarc_mli/doc/build/html/getting_started/getting_started.html) for details-->

***Q: Can I use MetaWare Development Tools Lite to pre-build embARC MLI library and ARC GNU to build example application?***  
A: No you cannot. <!--embARC MLI Library must be built by full version of MetaWare Development Tools. Binaries built with MWDT Lite are not compatible with ARC GNU Tools and full MetaWare Development Tools. Read the MWDT Lite documentation for details.-->

***Q: I can not build and run example application for my Synopsys board (EMSK, IoTDK, etc), what I shall do?***  
A: Don't do this until we come up with a proper recommendations. <!--If you build for Synopsys boards refer to documentation [embarc.org](https://embarc.org/projects/development-systems/) as a good starting point. -->
<!--You should also note that example applications support different configurations for pre trained models and thus memory requirements, not all configurations can be built and run on Synopsys boards due to memory limitations and HW capabilities, read example application readme for details. embARC MLI Library must be also pre built specifically for your board by MetaWare Development Tools. Please note that makefiles provided with examples are configured for IoTDK only if GNU tools are used. -->
