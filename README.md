EmbARC Machine Learning Inference Library
==================================================

This repository contains source code of embARC Machine Learning Inference Library (embARC MLI Lib),
examples and documentation.

## Release notes
----------------
1. Preliminary version 0.5
2. This release supports following functional primitives
	* 2D Convolution
	* 2D depthwise convolution
	* Fully Connected layer
	* Max and average pooling
	* LSTM, Basic RNN
	* Elementwise (add, sub, mul, min, max)
	* Data manipulation (concatanation, permute, 2D padding)
	* ReLU, Leaky ReLu, ReLu1, ReLu6
	* Softmax, Sigmoid, ThanH
3. Supported data layout CHW (Channel-Height-Width standard for Caffe)

## Package structure
--------------------
./bin                             - directory holder for embARC MLI library and samples binaries created during build  
./build                           - contains common build rules  
./doc                             - contains the API documentation of the embARC MLI library
./include                         - include files with API prototypes and types  
./lib/src                         - source code of embARC MLI Library  
./examples                        - source code of examples  
./examples/example_cifar10_caffe  - example illustrating implementation of CIFAR10 Caffe  
./examples/example_har_smartphone - example illustrating implementation of Human Activity Recognition  
./examples/auxilary               - source code of helper functions used for the examples  
./hw                              - contains HW templates (*.tcf files)   

## Building and quick start
---------------------------
By default embARC MLI Library can be build for [/hw/em9d.tcf](/hw/em9d.tcf) which is based on the standard EM9D Voice Audio template, 
defined in [MetaWare Development Tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware), with extended XY memory. embARC MLI Library can be also built for a specific 
EM or HS configuration.

Build requirements:
1. MetaWare Development tools 2018.12 or later

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
		

## Known Issues
---------------
None

