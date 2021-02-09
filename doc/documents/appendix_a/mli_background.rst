Appendix A – MLI Background
===========================

Introduction
------------

empty

.. _history_mli:
 
History of MLI
--------------

MLI was originally designed as an open source software library to enable 
machine learning on ARC EM and HS class processors.  From the original 
MLI Documentation [7], the stated goals were:

 - Deliver easy-to use SW library for fast and simple porting of NN 
   applications to ARCv2 processors

 - Support import of trained models from common Machine Learning Frameworks 

 - An efficient solution for small/middle size models inference using small 
   or very limited resources

 - Strengthen ARCv2 DSP positioning as low power inference engine 

MLI 1.0 was officially released in July 2019 and is available as open source 
in Github.  See here_ .

.. _here: https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases

The current API supports a “one function per layer” model, where application developers 
call a series of MLI APIs from their code, thus executing the layers of their graph one-by-one.  
Today, it’s the developer’s responsibility to manually “map” their graph from a 3rd-party 
framework (like TensorFlow, Caffe, etc) to MLI API calls.  This also involves preparing the 
input tensors of weights and activations.  Users are also required to manually quantize and 
create the input data structures in their code.  These steps are documented in the `User Guide`_ 
and via publicly available tutorials_. 

.. _User Guide: https://embarc.org/embarc_mli/doc/build/html/Examples_Tutorials/Examples_Tutorials.html

.. _tutorials: https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/mli_dev/examples/tutorial_emnist_tensorflow

Future of MLI
-------------

Going forward, there is a need to grow MLI beyond just a Machine Learning software library 
for EM and HS; Instead, the proposal is for MLI to become a standard.  This MLI Standard would 
be a strict API interface definition that could be implemented on any processor or ML accelerator 
hardware.  The reason for doing this is that if all ML library implementations for Synopsys 
processors/accelerators follow the same API convention, then front-end tools (like graph mappers, 
etc) can, in theory, easily target any processor with a single implementation.

To aid with understanding this concept, it might be useful to look at other projects that 
differentiate between specification and implementation.  A good example is the uITRON RTOS 
specification.  The specification appears at http://www.ertl.jp/ITRON/SPEC/FILE/mitron-400e.pdf.  
This specification defines the goals of the project, naming conventions, data formats, the API 
structure, etc, but does not associate itself with any specific RTOS implementation.  Other 
vendors have then taken this specification and created a specific RTOS implementation which follows 
all the rules defined in the RTOS spec – see ThreadX, for example 
(https://rtos.com/solutions/threadx/compatibility-for-uitron/).  A uITRON-compliant RTOS should be 
able to run any application which uses the uITRON API with little-to-no changes.  

Another well-known standard is the POSIX Threads (pthreads) library, where several implementations 
exist for multiple environments. In this case, the specific implementations target a specific Synopsys 
processor or accelerator. For the immediate future, the plan is to target ARC EM, HS, and VDSP 
(and VPX). In the future, accelerators like the DNN engine on future versions of the EV processor might 
be considered.

New Features in MLI 2.0
-----------------------

 - Added support for Synopsys ARC VPX5 processor

 - Continues to include support for ARC EM and HS processor families

 - Added new kernels (see :ref:`t_mli1_mli2_krn_diff`)

 - Integration with TensorFlow Lite for Microcontrollers (TFLM)

 - Data formats have been adapted
 
 - Supported Data layouts have been streamlined 
 
 - List of supported kernels has been extended
 
 - Some kernel functionality and interfaces have changed 
 
 - Data move functionality has been added
 
 - Support for additional platforms

Data Formats
~~~~~~~~~~~~

.. table:: MLI Data Formats Differences between MLI 1.0 and MLI 2.0
   :align: center
   :widths: auto
   
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | **Data Format** | **Format Name**          | How Used in MLI 1.0          | How Used in MLI 2.0                       | Comments                             |
   +=================+==========================+==============================+===========================================+======================================+
   | fx16            | 16-bit fixed point       | Used as main data format     | Used as main data format                  |                                      |
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | fx8             | 8-bit fixed point        | Used as main data format     | Only in conversion function               | In MLI2.0 this is replaced by sa8    |
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | sa32            | 32-bit signed asymmetric | Not available                | Used only for bias inputs to kernels      | Required to support TensorFlow Lite  |
   |                 |                          |                              |                                           | Micro                                |
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | sa8             | 8-bit signed asymmetric  | Not available                | Used as main 8-bit data format for        | Required to support TensorFlow Lite  |
   |                 |                          |                              | kernel inputs and outputs                 | Micro                                |
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | fp32            | 32-bit floating point    | Only in external conversion  | Only in conversion function as interface  |                                      |
   |                 |                          | function, not part of API    | between MLI and user code.                |                                      |
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+ 
..

fx8 has been replaced by sa8 because sa8 gives better accuracy (in most tensors a large part of the 
negative range is not used) and the performance drawback was estimated around 5% which is not enough
to justify the effort of supporting both data types.

For 16-bit, the accuracy benefit is relatively smaller (almost one extra bit on 16 bit has a smaller 
impact than almost one extra bit on 8bit) so there the tradeoff was made toward best performance.
fp16 has been considered, but for several reasons it was decided to not include it in the MLI 2.0 spec.
One of the reasons is that for inference, fp16 doesn't give significant better accuracy than fx16. 
(fx16 has a larger mantissa) but it can make a difference in area: on many platforms the FP is optional, 
whereas the int accumulators are always available. Other reasons are implementation cost and cross 
platform compatibility.
 
The sa8 type can also support per axis quantization. This means that for instance each channel can have 
a different zero point and scale factor. It depends on the kernel which of the input tensors can have per 
axis quantization.

Data Layout
~~~~~~~~~~~

It has been decided to support a single data layout because of ease of use and implementation cost 
reasons. The decision of the used data layout (HWCN) is based on the analysis of optimal vectorization 
for many graphs in a database. (N is the number of output channels).

.. table:: Supported Data Layouts: Differences between MLI 1.0 and MLI 2.0
   :align: center
   :widths: auto
   
   +-----------------+----------------------------------------------------+
   | **MLI Version** | **Supported Data Layouts**                         |
   +=================+====================================================+
   | 1.0             | CHW (fully optimized)                              |
   |                 | (N)HWC (only reference code)                       |
   +-----------------+----------------------------------------------------+
   | 1.1             | CHW (fully optimized)                              |
   |                 | (N)HWC (optimized in case of sa8   data format)    |
   +-----------------+----------------------------------------------------+
   | 2.0             | HWC(N)                                             |
   +-----------------+----------------------------------------------------+
..

.. note:: 
 
    That the layout is only relevant for a subset of functions. Most functions are layout-agnostic.
..

Kernels
~~~~~~~

.. _t_mli1_mli2_krn_diff:
.. table:: Supported kernels: Differences between MLI 1.0 and MLI 2.0
   :align: center
   :widths: auto
   
   +---------------------------------+-----------+----------------+
   | Kernel                          | MLI1.x    | MLI2.0         |
   +=================================+===========+================+
   | Conv2d                          | X         | X              |
   +---------------------------------+-----------+----------------+
   | Depthwise_conv2d                | X         | X              |
   +---------------------------------+-----------+----------------+
   | Transpose conv2d                |           | X              |
   +---------------------------------+-----------+----------------+
   | Group_conv2d                    |           | X              |
   +---------------------------------+-----------+----------------+
   | Avepool                         | X         | X              |
   +---------------------------------+-----------+----------------+
   | Maxpool                         | X         | X              |
   +---------------------------------+-----------+----------------+
   | Fully_connected                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | LSTM                            | X         | X              |
   +---------------------------------+-----------+----------------+
   | RNN                             | X         | X              |
   +---------------------------------+-----------+----------------+
   | GRU_cell (gated recurrent unit) |           | X              |
   +---------------------------------+-----------+----------------+
   | ReLu                            | X         | X              |
   +---------------------------------+-----------+----------------+
   | Leaky Relu                      | X         | X              |
   +---------------------------------+-----------+----------------+
   | Parametric ReLu                 |           | X              |
   +---------------------------------+-----------+----------------+
   | Sigm                            | X         | X              |
   +---------------------------------+-----------+----------------+
   | Tanh                            | X         | X              |
   +---------------------------------+-----------+----------------+
   | Softmax                         | X         | X              |
   +---------------------------------+-----------+----------------+
   | Elementwise_add                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | Elementwise_sub                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | Elementwise_mul                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | Elementwise_min                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | Elementwise_max                 | X         | X              |
   +---------------------------------+-----------+----------------+
   | Permute                         | X         | X              |
   +---------------------------------+-----------+----------------+
   | Concat                          | X         | Supported by   |
   |                                 |           | data move APIs |
   +---------------------------------+-----------+----------------+
   | Padding2d                       | X         | Supported by   |
   |                                 |           | data move APIs |
   +---------------------------------+-----------+----------------+
   | Argmax                          |           | X              |
   +---------------------------------+-----------+----------------+
..

Platforms
~~~~~~~~~

MLI 1.x supports EM and HS platforms and in MLI 2.0 support for VPX is added.
