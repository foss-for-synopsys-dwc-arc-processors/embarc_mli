Getting Started
---------------

   The MLI library provides a basic set of functions running on the ARC
   EM DSP and HS cores and is intended to solve typical machine learning
   tasks in a portable or low-power system. This library is designed
   with ease of use and low overhead in mind.

Package content
~~~~~~~~~~~~~~~

   This package consists of the following files and folders:

   +---------------------------------+-----------------------------------------+-----------------------+
   | **Folder**                      | **Files**                               | **Description**       |
   +=================================+=========================================+=======================+
   |                                 | ``Makefile``                            | Root makefile that    |
   |                                 |                                         | allows to build       |
   |                                 |                                         | library and the set   |
   |                                 |                                         | of example            |
   |                                 |                                         | applications.         |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **include/**                                                                                      |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``MLI_API.h``                           | High level header     |
   |                                 |                                         | used by application.  |
   |                                 |                                         | Includes all required |
   |                                 |                                         | headers for working   |
   |                                 |                                         | with library.         |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``MLI_types.h``                         | Header that           |
   |                                 |                                         | conglomerates all     |
   |                                 |                                         | public library        |
   |                                 |                                         | specific data types.  |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``MLI_config.h``                        | Configuration header  |
   |                                 |                                         | with definitions used |
   |                                 |                                         | for library           |
   |                                 |                                         | implementation        |
   |                                 |                                         | configurability.      |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``api``                                 | The subdirectory that |
   |                                 |                                         | contains the set of   |
   |                                 |                                         | low level public      |
   |                                 |                                         | headers declaring the |
   |                                 |                                         | API.                  |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **include/api**                                                                                   |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``MLI_helpers_API.h``                   | Header with           |
   |                                 |                                         | declarations for      |
   |                                 |                                         | helper functions      |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``MLI_kernels_API.h``                   | Header with           |
   |                                 |                                         | declarations for      |
   |                                 |                                         | kernel functions      |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``mli_krn_avepool_spec_i.h``            | Header with           |
   |                                 |                                         | declarations for      |
   |                                 |                                         | average pooling       |
   |                                 |                                         | specializations       |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``mli_krn_conv2d_spec_api.h``           | Header with           |
   |                                 |                                         | declarations for      |
   |                                 |                                         | convolution 2D        |
   |                                 |                                         | specializations       |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``mli_krn_depthwise_conv2d_spec_api.h`` | Header with           |
   |                                 |                                         | declarations for      |
   |                                 |                                         | depth-wise            |
   |                                 |                                         | convolution           |
   |                                 |                                         | specializations       |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``mli_krn_maxpool_spec_api.h``          | Header with           |
   |                                 |                                         | declarations for max  |
   |                                 |                                         | pooling               |
   |                                 |                                         | specializations       |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **build/**                                                                                        |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``rules.mk``                            | The common makefile   |
   |                                 |                                         | that declares the     |
   |                                 |                                         | generic build targets |
   |                                 |                                         | and rules.            |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **lib/**                                                                                          |
   +---------------------------------+-----------------------------------------+-----------------------+
   | lib/make/                       | ``Makefile``                            | Library makefile.     |
   +---------------------------------+-----------------------------------------+-----------------------+
   | lib/src/helpers/                |                                         | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | helper functions:     |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **lib/kernels/**                                                                                  |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``common/``                             | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | common kernels        |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``convolution/``                        | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | convolution kernels   |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``elementwise/``                        | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | elementwise kernels   |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``pooling/``                            | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | pooling kernels       |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``data_manip/``                         | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | data manipulation     |
   |                                 |                                         | kernels               |
   +---------------------------------+-----------------------------------------+-----------------------+
   |                                 | ``transform/``                          | Directory containing  |
   |                                 |                                         | implementation of     |
   |                                 |                                         | transform kernels     |
   +---------------------------------+-----------------------------------------+-----------------------+
   | lib/private/                    |                                         | Directory containing  |
   |                                 |                                         | internal library      |
   |                                 |                                         | specific functions    |
   |                                 |                                         | and data description  |
   +---------------------------------+-----------------------------------------+-----------------------+
   | **examples/**                                                                                     |
   +---------------------------------+-----------------------------------------+-----------------------+
   | examples/Example_CIF            |                                         | Convolutional Neural  |
   | R10_Caffe                       |                                         | Network example for   |
   |                                 |                                         | CIFAR-10 dataset (See |
   |                                 |                                         | entry [6] of          |
   |                                 |                                         | :ref:`refs`)          |
   |                                 |                                         | Based on Caffe        |
   |                                 |                                         | standard example (see |
   |                                 |                                         | entry [7] of          |
   |                                 |                                         | :ref:`refs`)          |
   +---------------------------------+-----------------------------------------+-----------------------+
   | examples/Example_HAR_Smartphone |                                         | LSTM Human activity   |
   |                                 |                                         | recognition example.  |
   |                                 |                                         | Based on open source  |
   |                                 |                                         | project by Guillaume  |
   |                                 |                                         | Chevalier (See entry  |
   |                                 |                                         | [8] of :ref:`refs`)   |
   |                                 |                                         | for UCI HAR dataset   |
   |                                 |                                         | (See entry [9] of     |
   |                                 |                                         | :ref:`refs`).         |
   +---------------------------------+-----------------------------------------+-----------------------+
   | examples/auxiliary              |                                         | Common helper code    |
   |                                 |                                         | that is used by the   |
   |                                 |                                         | examples.             |
   +---------------------------------+-----------------------------------------+-----------------------+

Build Process
~~~~~~~~~~~~~

   To verify the build process, run the following root makefile command
   to rebuild the project from scratch and place the output ELF and
   static library files into the bin directory.

   ``gmake all``

   After the build process completes, the output files are found inside
   the bin directory created by the make tool:

.. code:: c

   c:\mli\bin>dir /b *.elf *.a
   Example_CIFAR10_Caffe.elf
   Example_HAR_Smartphone.elf
       libmli.a

..
   
   Root makefile supports the configuration option ``TCF_FILE`` to set the
   required TCF file to use for the code compilation. The default TCF
   file hard-coded in the project is em7d_voice_audio provided as a part
   of MetaWare Development Toolkit package. Note that makefile requires the full path to
   the TCF in case of using the custom file.

   ``gmake all TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf``

..
   
   The project build system also allows to build the library and
   examples separately.

   It supports the common set of optional parameters to configure the
   target platform and output/intermediate files placement:

   +-----------------------------------+-----------------------------------+
   |    **Parameter**                  |    **Description**                |
   +===================================+===================================+
   |    ``TCF_FILE``                   |    The name of the TCF provided   |
   |                                   |    as a part of the MWDT package  |
   |                                   |    or a full path of the custom   |
   |                                   |    TCF                            |
   +-----------------------------------+-----------------------------------+
   |    ``LIBRARY_DIR``                |    Target directory to create the |
   |                                   |    static library                 |
   +-----------------------------------+-----------------------------------+
   |   ``BUILD_DIR``                   |    Target directory to store the  |
   |                                   |    intermediate files (object     |
   |                                   |    files, and so on).             |
   +-----------------------------------+-----------------------------------+
   |    ``OUT_DIR``                    |    Target directory to create the |
   |                                   |    example ELF                    |
   +-----------------------------------+-----------------------------------+
   |    ``OUT_NAME``                   |    Custom file name of the        |
   |                                   |    library or example             |
   +-----------------------------------+-----------------------------------+
   |    ``LCF_FILE``                   |    Custom linker script file to   |
   |                                   |    use to build the example. Dy   |
   |                                   |    default, the linker script is  |
   |                                   |    obtained from the TCF.         |
   +-----------------------------------+-----------------------------------+
   |    ``EXT_CFLAGS``                 |    Additional C compiler flags to |
   |                                   |    use                            |
   +-----------------------------------+-----------------------------------+

\

   Use the library makefile located in ``lib/make`` directory to create the
   custom library build.

Examples:
~~~~~~~~~

-  To build the debug version of the library using the custom TCF and
   storing the output file in the directory outside the project, use the
   following commands:

.. code:: c

   cd lib\make
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf 
   EXT_CFLAGS=-g LIBRARY_DIR=C:\bin

..
   
   Example applications also provide the separate makefiles to make the
   customized applications build.

-  To build the debug version of the CIFAR10_Caffe example application
   using the custom TCF, use the following commands:

.. code:: c

   cd examples\Example_CIFAR10_Caffe
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf 
   EXT_CFLAGS=-g

..

   Example application makefiles support the target run to execute the
   application using the NSIM simulator. Note that this target
   requires the TCF name to be provided to setup the simulation
   environment.

-  To build ``CIFAR10_Caffe`` example application using the custom TCF and
   starting the simulation, use the following commands:

.. code:: c

   cd examples\Example_CIFAR10_Caffe
   gmake TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf
   gmake run TCF_FILE=C:\ARC\Projects\my_project\build\tool_config\arc.tcf

..