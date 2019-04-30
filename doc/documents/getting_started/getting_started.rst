Getting Started
===============

The MLI library provides a basic set of functions running on the ARC
EM DSP and HS cores. It is intended to solve typical machine learning
tasks in a portable or low-power system. This library is designed
with ease of use and low overhead in mind.
   
.. _bld_lib:

Build Library
-------------

By default, the embARC MLI Library can be built for ``/hw/em9d.tcf`` that is based on 
the standard EM9D Voice Audio template, defined in MetaWare Development Tools with extended 
XY memory. The embARC MLI Library can be also built for a specific EM or HS configuration.

Build requirements:

MetaWare Development tools 2018.12 or later

To verify the build process, run the following root makefile command
to rebuild the project from scratch including  ELF for examples and static library:

.. code-block:: console

   gmake all
   
..

After the build process completes, the output files are found inside
the bin directory created by the make tool:

On Windows:
   
.. code-block:: console

   dir /b /s \*.elf \*.a
..

On Linux:
   
.. code-block:: console

   find -name ‘\*.a’ -o -name ‘\*.elf’
..

Expected console output on windows is next:: 

.\bin\libmli.a
.\examples\example_cifar10_caffe\bin\example_cifar10_caffe.elf
.\examples\example_har_smartphone\bin\example_har_smartphone.elf

The project build system allows to build the library and examples separately. 
Library and each example have a separate makefile which uses common rules from ``build/rules.mk``.
To build the embARC MLI library defined by a TCF file, use the following command:

.. code:: c

   gmake all TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf

..
   
The Build system supports a common set of optional parameters to configure the target platform, 
output/intermediate files placement, and provides extra parameters to build tool or application.
   
.. table:: Optional Parameters for Target Platform
   :widths: auto
   
   +-------------------------+---------------------------------------------+
   |    **Parameter**        |    **Description**                          |
   +=========================+=============================================+
   |    ``TCF_FILE``         |    The name of the TCF provided             |
   |                         |    as a part of the MWDT package            |
   |                         |    or a full path of the custom             |
   |                         |    TCF                                      |
   +-------------------------+---------------------------------------------+
   |    ``LIBRARY_DIR``      |    Target directory to create the           |
   |                         |    static library                           |
   +-------------------------+---------------------------------------------+
   |   ``BUILD_DIR``         |    Target directory to store the            |
   |                         |    intermediate files (object               |
   |                         |    files, and so on).                       |
   +-------------------------+---------------------------------------------+
   |    ``OUT_DIR``          |    Target directory to create the           |
   |                         |    example ELF                              |
   +-------------------------+---------------------------------------------+
   |    ``OUT_NAME``         |    Custom file name of the                  |
   |                         |    library or example                       |
   +-------------------------+---------------------------------------------+
   |    ``LCF_FILE``         |    Custom linker script file to             |
   |                         |    use to build the example. Dy             |
   |                         |    default, the linker script is            |
   |                         |    obtained from the TCF.                   |
   +-------------------------+---------------------------------------------+
   |    ``EXT_CFLAGS``       |    Additional C compiler flags to           |
   |                         |    use                                      |
   +-------------------------+---------------------------------------------+
   |     ``RUN_ARGS``        |    Application command line arguments       |
   |                         |                                             |
   +-------------------------+---------------------------------------------+

.. _bld_run_ex:
   
Build and Run Examples
----------------------

To build the debug version of the library using the custom TCF and
to store the output file in the directory outside the project, use the
following commands:

.. code:: console

   cd lib\make
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf EXT_CFLAGS=-g LIBRARY_DIR=C:\bin

..
   
Example applications also provide the separate makefiles to make the
customized applications build.

To build the debug version of the ``cifar10_caffe`` example application
using the custom TCF, use the following commands:

.. code:: console

   cd examples\example_cifar10_caffe
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf EXT_CFLAGS=-g

..

Example application makefiles support the target run to execute the
application using the NSIM simulator. Note that this target
requires the TCF name to be provided to setup the simulation
environment.

To build ``cifar10_caffe`` example application using the custom TCF and
starting the simulation, use the following commands:

.. code:: console

   cd examples\example_cifar10_caffe
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf
   gmake run TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf

..

.. _pkg_struct:

Project Structure
-----------------

The repo is organized as follows:

* ``./build/``: Contains common build rules.
* ``./build/rules.mk``: The common makefile that declares the generic build targets and rules.
* ``./doc/``: Contains sources of embARC MLI library documentation (what you are reading now).
* ``./include/``:  Include files with API prototypes and types. Subject for more attention.
* ``./include/mli_api.h``: High level header used by application. Includes all required headers for working with library.
* ``./include/mli_types.h``: Header that conglomerates all public library specific data types.
* ``./include/mli_config.h``: Configuration header with definitions used for library implementation configu-rability.
* ``./include/api/``: The subdirectory that contains the set of low level public headers declaring the API.
* ``./include/api/mli_helpers_api.h``: Header with declarations for helper functions
* ``./include/api/mli_kernels_api.h``: Header with declarations for kernel functions
* ``./include/api/mli_krn_avepool_spec_api.h``: Header with declarations for average pooling special-izations
* ``./include/api/mli_krn_conv2d_spec_api.h``: Header with declarations for convolution 2D speciali-zations
* ``./include/api/mli_krn_depthwise_conv2d_spec_api.h``: Header with declarations for depth-wise convolution specializations
* ``./include/api/mli_krn_maxpool_spec_api.h``: Header with declarations for max pooling specializa-tions
* ``./lib/``: Source code and build scripts of embARC MLI Library 
* ``./examples/``: Source code of examples
* ``./examples/example_cifar10_caffe/``: Convolutional Neural Network example for CIFAR-10 dataset. 
* ``./examples/example_har_smartphone/``: LSTM based Human activity recognition example.
* ``./examples/auxiliary/``: Common helper code that is used by the examples.
* ``./examples/prebuilt/``: Library to be filled with prebuilt MLI Library for working with examples via ARC GNU tools.
* ``./hw/``: Contains HW templates (\*.tcf files)
* ``./Makefile``: Root makefile that allows to build or clean library and the set of example applications.
* ``./README.md``: Short description and quick start instructions for embARC MLI Library
* ``./LICENSE``: License notes


After you have built library, the following entities will appear in the structure:

* ``./obj/``: Directory holder for library object and dependency files created during build
* ``./bin/``: Directory holder for embARC MLI library binaries created during build
* ``./bin/libmli.a``: embARC MLI library archive file (static library)


