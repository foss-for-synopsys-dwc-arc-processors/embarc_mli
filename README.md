embARC Machine Learning Inference Library
==================================================

This repository contains source code of embARC Machine Learning Inference Library (embARC MLI Library), 
documentation and examples. Read the documentation at [embarc.org](https://embarc.org/embarc_mli).

## Release notes
----------------
1. Version 1.0
2. This release supports following functional primitives
	* 2D Convolution
	* 2D depthwise convolution
	* Fully Connected layer
	* Max and average pooling
	* LSTM, Basic RNN
	* Elementwise (add, sub, mul, min, max)
	* Data manipulation (concatanation, permute, 2D padding)
	* ReLU, Leaky ReLu, ReLu1, ReLu6
	* Softmax, Sigmoid, TanH
3. Supported data layout CHW (Channel-Height-Width standard for Caffe)

## Package structure
--------------------
/bin                             		- directory holder for embARC MLI library and samples binaries created during build  
./build                           		- contains common build rules  
./doc                             		- contains the API documentation of the embARC MLI library  
./include                         		- include files with API prototypes and types  
./lib/src                         		- source code of embARC MLI Library  
./examples                        		- source code of examples  
./examples/example_cifar10_caffe  		- example illustrating implementation of CIFAR10 Caffe  
./examples/example_har_smartphone 		- example illustrating implementation of Human Activity Recognition  
./examples/auxilary               		- source code of helper functions used for the examples  
./examples/tutorial_cifar10_caffe_deployment	- model deployment tutorial for Caffe and CIFAR10  
./hw                              		- contains HW templates (*.tcf files)   

## Building and quick start
---------------------------
By default embARC MLI Library can be build for [/hw/em9d.tcf](/hw/em9d.tcf) which is based on the standard EM9D Voice Audio template, 
defined in [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware), with extended XY memory. embARC MLI Library can be also built for a specific 
EM or HS configuration.

Build requirements:
1. MetaWare Development tools 2019.03-1 or later

Building of embARC MLI library	
1. Open command line and change working directory to './lib/make/'      
2. Start building
	'gmake TCF_FILE=../../hw/em9d.tcf'

## Building and running [CIFAR10 Caffe example](examples/example_cifar10_caffe/README.md)
---------------------------------------------
1. Open command line and change working directory to './examples/example_cifar10_caffe/'

2. Build CIFAR10 example
	'gmake TCF_FILE=../../hw/em9d.tcf'

3. Run CIFAR10 example
	'gmake run TCF_FILE=../../hw/em9d.tcf'

4. Result Quality shall be "S/N=3638.6     (71.2 db)"

## Building and running [Human Activity Recognition example](examples/example_har_smartphone/README.md)
----------------------------------------------------------
1. Open command line and change working directory to './examples/example_har_smartphone'

2. Clean old build artifacts of the application (this will not clean the lib)
	'gmake clean'

3. Build HAR example.
	'gmake TCF_FILE=../../hw/em9d.tcf'

4. Run HAR example.
    'gmake run TCF_FILE=../../hw/em9d.tcf'

5. Result Quality shall be "S/N=1823.9     (65.2 db)"
		
## Optimizations for code size
------------------------------
By default the embARC MLI Library is build for optimal speed. If code size needs to be reduced, there are two things that can be done:
1. For convolution and pooling layers there are specialized funtions for specific kernel sizes, they are called by a wrapper functions based on the parameters.
These parameters are compile time constant in the application, so the application can directly call the specialized functions. This will reduce over all code size.
Please be aware that the list of specializations is not guaranteed to be backwards compatible between releases.

2. Use a different optimization mode when calling the makefile. OPTMODE=size will optimize for size. default is OPTMODE=speed
	'gmake TCF_FILE=../../hw/em9d.tcf OPTMODE=size'

## Known Issues
---------------
1. Optimal performance for 8-bit data requires version of MetaWare Development Tools 2019.06 or later

## Frequently Asked Questions
---------------

***Q: Can I use ARC GNU tools to build embARC MLI library?***  
A: No you cannot. embARC MLI Library must be built by MetaWare Development Tools only. Read the documentation at [embarc.org]( https://embarc.org/embarc_mli/doc/build/html/getting_started/getting_started.html#build-library) for details

***Q: Can I use MetaWare Development Tools Lite to pre-build embARC MLI library and ARC GNU to build example application?***  
A: No you cannot. embARC MLI Library must be built by full version of MetaWare Development Tools. Binaries built with MWDT Lite are not compatible with ARC GNU Tools and full MetaWare Development Tools. Read the MWDT Lite documentation for details.

***Q: I can not build and run example application for my Synopsys board (EMSK, IoTDK, etc), what I shall do?***  
A: If you build for Synopsys boards refer to documentation [embarc.org](https://embarc.org/platforms.html) as a good starting point. 
You should also note that example applications support different configurations for pre trained models and thus memory requirements, not all configurations can be built and run on Synopsys boards due to memory limitations and HW capabilities, read example application readme for details. embARC MLI Library must be also pre built specifically for your board by MetaWare Development Tools. Please note that makefiles provided with examples are configured for IoTDK only if GNU tools are used.
