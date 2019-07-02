Face Detect Example
==============================================
Example shows very basic implementation of the classic object detection via sliding window paradigm.
Small binary NN classifier for 36x36 grayscale images outputs positive decision for the images of face, and negative decision on other images. To process bigger image pyramid scaling and sliding is organized. 
Such approach still useful for deeply embedded applications as a compact and efficient way of triggering a bigger job. Activation function and the Layer 2 are quite unspecific kind of layers and was implemented in the research activity. 
MLI provides useful data manipulation and helper operations for implementation of such layers.


Quick Start
--------------

Example supports building with [MetaWare Development tools](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware) and [ARC GNU toolchain](https://www.synopsys.com/dw/ipdir.php?ds=sw_jtag_gnu) and running with MetaWare Debuger on [nSim simulator](https://www.synopsys.com/dw/ipdir.php?ds=sim_nSIM).

### Build with MetaWare Development tools

    Build requirements:
        - MetaWare Development tools version 2018.12 or higher
        - gmake

Here we will consider building for [/hw/em9d.tcf](/hw/em9d.tcf) template. This template is a default template for this example. Other templated can be also used. 

0. embARC MLI Library must be built for required hardware configuration first. See [embARC MLI Library building and quick start](/README.md#building-and-quick-start).

1. Open command line and change working directory to `./examples/example_face_detect`

2. Clean previous build artifacts (optional)

       gmake clean

3. Build example 

       gmake TCF_FILE=../../hw/em9d.tcf 

### Run example with MetaWare Debuger on nSim simulator.

Example application requires path to a BMP file of 80x60 resolution and 24 bit depth (RGB) as an input parameter.

       gmake run TCF_FILE=../../hw/em9d.tcf RUN_ARGS=grace_hopper.bmp

Application will create `result.bmp` file in the working directory. It's a 'grayed' version of input file with framed faces which was found in run. Expected console output is next: 

    Detection step #0
     Found a face at ([X:22, Y:17]; [X:58, Y:53])
    Detection step #1
    Detection step #2
    Detection step #3
    Detection step #4
     Found a face at ([X:13, Y:11]; [X:55, Y:53])
    Detection step #5
    Detection step #6
    Detection step #7
    Detection step #8


### Build with ARC GNU toolchain

Here we will consider building with ARC GNU toolchain. As a platform for the assembly, we use the [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) from [the embARC Open Software Platform (OSP)](https://embarc.org/embarc_osp/doc/build/html/introduction/introduction.html#)

    Build requirements:
        - ARC GNU toolchain version 2018.09 or higher
        - embARC MLI Library prebuilt with MetaWare Development tools for IoT Devkit hardware configuration
        - gmake

0. Prebuilt embARC MLI Library  must be copyied into the `./examples/prebuilt` folder.

1. Open command line and change working directory to `./examples/example_face_detect`

2. Clean previous build artifacts (optional)

        gmake TOOLCHAIN=gnu clean

3. Build example

        gmake TOOLCHAIN=gnu

   Notes: IoT Devkit hardware configuration is specifed in Makefile. Additionally used memory.x linkscript file for GNU linker. 

### Run example with MetaWare Debuger on nSim simulator.

    Run requirements:
    - MetaWare Development tools version 2018.12 or higher
    - arcem9d.tcf file with hardware configuration of IoT Devkit for setup nSim.

0. Copy the [arcem9d.tcf](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_osp/blob/master/board/iotdk/configs/10/tcf/arcem9d.tcf) file into example folder.

1. Run example 

        gmake run TOOLCHAIN=gnu TCF_FILE=arcem9d.tcf

    Notes: Example built by ARC GNU tools is run using mdb_com_gnu script file. Modify this file to customize the example input argument (path to the image).

### Run example without MetaWare Development tools

See documentation on [IoT Devkit](https://embarc.org/embarc_osp/doc/build/html/board/iotdk.html) on how to run executable built with [ARC GNU](https://embarc.org/toolchain/index.html) and [ARC open source development tools](https://embarc.org/embarc_osp/doc/build/html/index.html) on IoT Devkit.

Example Structure
--------------------
Structure of example application may be divided logically on three parts:

* **Application.** Implements resources allocation, reading and writing BMP files and invoking face search by pre-defined sliding scheme:
   * bmp_file_io.c
   * bmp_file_io.h
   * main.c
* **Sliding window and rescaling code.** Various helper functions to scale input image and slide trigger classifiyer over it.
   * sliding_scan.c
   * sliding_scan.h
* **Inference Module.** Uses embARC MLI Library to process input according to pre-defined graph. All model related constants are pre-defined and model coefficients is declared in the separate compile unit 
   * face_trigger_constants.h
   * face_trigger_model.c
   * face_trigger_model.h
   

Example structure contains test image of [Grace Hopper](https://en.wikipedia.org/wiki/Grace_Hopper).

Example structure also contains auxiliary files for development tools:
 * arcem9d.lcf - linkscript file for MetaWare linker.
 * memory.x    - linkscript file for GNU linker.
 * mdb_com_gnu - command script file for MetaWare Debugger.


Data Memory Requirements
----------------------------

Example uses statically allocated memory for model weights and intermediate results (activations) and structures. For images on the application level (outside of the model) 
example allocates memory dynamically befor processing.

|                      Data                         |         Size          |
| :-----------------------------------------------: | :-------------------: |
| Weights (*.mli_model* section)                    |  3056 bytes           |
| Activations (*.Xdata* and *.Ydata* sections)      |  5664 bytes           |
| Structures (*.mli_data* section)                  |  480 bytes            |
| Images (part of heap)                             |  14400 bytes          |

Application code size depends on target hardware configuration and compilation flags. MLI Library code is wrapped into mli_lib section.

References
----------------------------
> P. Viola and M. Jones. Rapid object detection using a boosted cascade of simple features. In CVPR, 2001

> R. Vaillant, C. Monrocq, and Y. LeCun. Original approach for the localisation of objects in images. IEE Proc. on Vision, Image, and Signal Processing, 1994.