.. ML_RST documentation master file, created by
   sphinx-quickstart on Fri Feb 15 10:54:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Definitions 
===========

.. _Definitions:
.. table:: Definitions
   :widths: auto
   
   +-----------------------------------+-----------------------------------+
   | **Term**                          | **Definition**                    |
   +===================================+===================================+
   | AGU                               | Address Generation Unit           |
   +-----------------------------------+-----------------------------------+
   | ARCv2DSP                          | Synopsys DesignWare® ARC®         |
   |                                   | Processors family of 32-bit CPUs  |
   +-----------------------------------+-----------------------------------+
   | ARC EMxD                          | Family of 32 bit ARC processor    |
   |                                   | cores. Single core, 3-step        |
   |                                   | pipeline, ARCv2DSP                |
   +-----------------------------------+-----------------------------------+
   | ARC HS4xD                         | Family of 32 bit ARC processor    |
   |                                   | cores. Multi core, dual issue,    |
   |                                   | 10-step pipeline, ARCv2DSP        |
   +-----------------------------------+-----------------------------------+
   | CHW                               | Channel-Height-Width data layout  |
   +-----------------------------------+-----------------------------------+
   | DMA                               | Direct memory Access              |
   +-----------------------------------+-----------------------------------+
   | DSP                               | Digital Signal Processor          |
   +-----------------------------------+-----------------------------------+
   | HWC                               | Height-Width-Channel data layout  |
   +-----------------------------------+-----------------------------------+
   | IoT                               | Internet Of Things                |
   +-----------------------------------+-----------------------------------+
   | MAC                               | Multiple Accumulate               |
   +-----------------------------------+-----------------------------------+
   | MWDT                              | MetaWare Development Tool set     |
   +-----------------------------------+-----------------------------------+
   | ML                                | Machine Learning                  |
   +-----------------------------------+-----------------------------------+
   | MLI Library                       | Machine Learning Inference        |
   |                                   | Library                           |
   +-----------------------------------+-----------------------------------+
   | MLI kernel                        | Basic operation on tensors in ML  |
   |                                   | model, provided by MLI Library as |
   |                                   | C-style API function. Typically   |
   |                                   | does not imply any intermediate   |
   |                                   | copying.                          |
   +-----------------------------------+-----------------------------------+
   | NN                                | Neural Network                    |
   +-----------------------------------+-----------------------------------+
   | Primitive                         | Basic ML functionality            |
   |                                   | implemented as MLI Kernel or MLI  |
   |                                   | Layer (Convolution 2D, Fully      |
   |                                   | connected, and so on)             |
   +-----------------------------------+-----------------------------------+
   | ReLU                              | Rectified Linear Unit             |
   +-----------------------------------+-----------------------------------+
   | TCF                               | Tool Configuration File. Hold     |
   |                                   | information about ARC processor   |
   |                                   | build configuration and           |
   |                                   | extensions.                       |
   +-----------------------------------+-----------------------------------+
   | Tensor                            | Object that contains binary data  |
   |                                   | and its complete description,     |
   |                                   | such as dimensions, element data  |
   |                                   | type, and so on.                  |
   +-----------------------------------+-----------------------------------+

\   


Revision history
================

   +-----------------------+-----------------------+-----------------------+
   | **Date**              | **Version**           | **Comment**           |
   +=======================+=======================+=======================+
   | 15-June-2018          | 0.1                   | Draft for initial     |
   |                       |                       | review, incorporating |
   |                       |                       | list of functions of  |
   |                       |                       | interest for primary  |
   |                       |                       | customer              |
   +-----------------------+-----------------------+-----------------------+
   | 6-Dec-2018            | 0.4                   | First version for     |
   |                       |                       | wide review with      |
   |                       |                       | complete set of main  |
   |                       |                       | kernels.              |
   +-----------------------+-----------------------+-----------------------+
   | 15-Jan-2019           | 0.6                   | Changes of document   |
   |                       |                       | and API according to  |
   |                       |                       | feedback of first     |
   |                       |                       | review                |
   +-----------------------+-----------------------+-----------------------+

