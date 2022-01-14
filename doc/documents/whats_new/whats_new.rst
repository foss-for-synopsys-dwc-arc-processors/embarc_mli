What's New in MLI 2.0 ?
=======================

New Features in MLI 2.0
-----------------------

 - Added support for Synopsys ARC VPX5 processor

 - Continues to include support for ARC EM and HS processor families

 - Added new kernels (see :ref:`t_mli1_mli2_krn_diff`)

 - Integration with TensorFlow Lite for Microcontrollers (TFLM)

 - Data formats have been adapted
 
 - Supported Data layouts have been streamlined 
 
 - Some kernel functionality and interfaces have changed 
 
 - Data move functionality has been added
 
 - Support for additional platforms
 
Data Formats
------------

.. tabularcolumns:: |\Y{0.15}|\Y{0.15}|\Y{0.2}|\Y{0.3}|\Y{0.2}|

.. table:: MLI Data Formats Differences Between MLI 1.0 and MLI 2.0
   :align: center
   :widths: auto
   
   +-----------------+--------------------------+------------------------------+-------------------------------------------+--------------------------------------+
   | **Data Format** | **Format Name**          | **How Used in MLI 1.0**      | **How Used in MLI 2.0**                   | **Comments**                         |
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

fx8 has been replaced by sa8 because sa8 gives better accuracy (in most tensors, a large part of the 
negative range is not used).

For 16-bit design, the accuracy benefit is relatively lesser (almost one extra bit has a lesser 
impact on 16-bit design than that on an 8-bit design). 
 
The sa8 type can also support per axis quantization. This means that for instance each channel can have 
a different zero point and scale factor. It depends on the kernel which of the input tensors can have per 
axis quantization.

Data Layout
-----------

MLI 2.0 supports the HWCN data layout.

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
-------

.. _t_mli1_mli2_krn_diff:
.. table:: Supported kernels: Differences between MLI 1.0 and MLI 2.0
   :align: center
   :widths: auto
   
   +---------------------------------+------------+----------------+
   | **Kernel**                      | **MLI1.x** | **MLI2.0**     |
   +=================================+============+================+
   | Conv2d                          | X          | X              |
   +---------------------------------+------------+----------------+
   | Depthwise_conv2d                | X          | X              |
   +---------------------------------+------------+----------------+
   | Transpose conv2d                |            | X              |
   +---------------------------------+------------+----------------+
   | Group_conv2d                    |            | X              |
   +---------------------------------+------------+----------------+
   | Avepool                         | X          | X              |
   +---------------------------------+------------+----------------+
   | Maxpool                         | X          | X              |
   +---------------------------------+------------+----------------+
   | Fully_connected                 | X          | X              |
   +---------------------------------+------------+----------------+
   | LSTM                            | X          | X              |
   +---------------------------------+------------+----------------+
   | RNN                             | X          | X              |
   +---------------------------------+------------+----------------+
   | GRU_cell (gated recurrent unit) |            | X              |
   +---------------------------------+------------+----------------+
   | ReLu                            | X          | X              |
   +---------------------------------+------------+----------------+
   | Leaky Relu                      | X          | X              |
   +---------------------------------+------------+----------------+
   | Parametric ReLu                 |            | X              |
   +---------------------------------+------------+----------------+
   | Sigm                            | X          | X              |
   +---------------------------------+------------+----------------+
   | Tanh                            | X          | X              |
   +---------------------------------+------------+----------------+
   | Softmax                         | X          | X              |
   +---------------------------------+------------+----------------+
   | Elementwise_add                 | X          | X              |
   +---------------------------------+------------+----------------+
   | Elementwise_sub                 | X          | X              |
   +---------------------------------+------------+----------------+
   | Elementwise_mul                 | X          | X              |
   +---------------------------------+------------+----------------+
   | Elementwise_min                 | X          | X              |
   +---------------------------------+------------+----------------+
   | Elementwise_max                 | X          | X              |
   +---------------------------------+------------+----------------+
   | Permute                         | X          | X              |
   +---------------------------------+------------+----------------+
   | Concat                          | X          | Supported by   |
   |                                 |            | data move APIs |
   +---------------------------------+------------+----------------+
   | Padding2d                       | X          | Supported by   |
   |                                 |            | data move APIs |
   +---------------------------------+------------+----------------+
   | Argmax                          |            | X              |
   +---------------------------------+------------+----------------+
..

Platforms
---------

MLI 2.0 supports EM, HS, and VPX platforms.


.. note::
   The pre-built documentation of other embARC MLI library versions can be found 
   on the `release page <https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases>`_ 
   in the corisponidng assets list. To use it download *embARC_MLI_\<version\>_API_Doc.zip* file,
   unzip the archive and open `<unzipped_archive_root>/html/index.html` file in a browser.
..
   