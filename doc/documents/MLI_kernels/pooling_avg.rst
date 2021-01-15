.. _avg_pool:
   
Average-Pooling
~~~~~~~~~~~~~~~

.. image:: ../images/image109.png
   :align: center
   :alt: Average Pooling Operation Example
   
..

This kernel implements an average-pooling operation. Input and output
feature maps are three-dimensional tensor even if one of dimensions is
equal to 1. Input shape includes height, width, and depth (channels)
according to used data layout (see :ref:`data_muldim`).

Each input channel is considered independently, which means that
analysis window includes only neighbor points of the channel. For
each window, average value over all considered ponts is defined as
the output value. Window size (or kernel size for convolution layers)
is defined in configuration structure according to kernel_width and
kernel_height values. Window positioning and moving is performed
according to stride and padding parameters. This logic is similar to
convolution 2D operation (see :ref:`cnvl_2d`).

Pooling primitive does not analyze an area smaller than kernel size
(typically, this occurs on the right and bottom borders). In this
case, ensure that you set padding parameters explicitly in order not
to miss valid border values. Padded values do not participate in the
calculations. So when a fragment includes padded values, only the
existing values are analyzed (this also implies reducing of divider
for average calculation).

To use padding in Caffe, use Padding2D (see :ref:`pad_2d`) primitive
before pooling.

.. caution::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.                   
   
.. _function-configuration-structure-3:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Average pooling kernel shares configuration structure with max
pooling kernel. For more information see :ref:`fn_conf_str_max_pool`
section of Max-pooling function.

.. _general-api-3:

Kernel Interface
^^^^^^^^^^^^^^^^

Interface of all specializations are the same and described as follows.
   
Prototype
'''''''''

.. code:: c                           
                                      
 mli_status  mli_krn_avepool_<layout>_
 <data_type>[_specialization](        
    const mli_tensor *in,             
    const mli_pool_cfg *cfg,       
    mli_tensor *out);                    
..
	
Parameters
''''''''''

.. table:: Kernel Interface Parameters
   :widths: 20,130
   
   +-----------------------+-----------------------+
   |  **Parameters**       | **Description**       |
   +-----------------------+-----------------------+
   |                       |                       |
   |                       |                       |
   |  ``in``               | [IN] Pointer to input |
   |                       | feature map tensor    |
   +-----------------------+-----------------------+
   |                       |                       |
   |                       |                       |
   |  ``cfg``              | [IN] Pointer to       |
   |                       | pooling parameters    |
   |                       | structure             |
   +-----------------------+-----------------------+
   |                       |                       |
   |                       |                       |
   |  ``out``              | [OUT] Pointer to      |
   |                       | output feature map    |
   |                       | tensor. Result is     |
   |                       | stored here           |
   +-----------------------+-----------------------+
   
.. _function-specializations-3:

Function Specializations
^^^^^^^^^^^^^^^^^^^^^^^^

There are about 80 specializations for the primitive assuming various
combinations of inputs parameters. Average-pooling primitive follows
the naming convention for specializations (see :ref:`spec_fns`
). The header file ``mli_krn_avepool_spec_api.h`` contains
declarations of all specializations for the primitive.

:ref:`Non-spec_funct_avg_pool` contains only non-specialized functions.

.. _Non-spec_funct_avg_pool:
.. table:: Non-specialized Functions
   :widths: 20,130
   
   +-------------------------------------+-----------------------------------+
   | **Function**                        | **Description**                   |
   +=====================================+===================================+
   ||                          *CHW Data Layout*                             |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx8``         | Switching function (see           |
   |                                     | :ref:`fns`); 8bit FX tensors;     |
   |                                     | Delegates calculations to         |
   |                                     | suitable specialization or        |
   |                                     | generic function                  |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx16``        | Switching function (see           |
   |                                     | :ref:`fns`); 16bit FX tensors;    |
   |                                     | Delegates calculations to         |
   |                                     | suitable specialization or        |
   |                                     | generic function                  |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx8_generic`` | General function; 8bit FX tensors |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx16_generic``| General function; 16bit FX        |
   |                                     | tensors                           |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx8_[spec]``  | Specialization function*; 8bit FX |
   |                                     | tensors                           |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_chw_fx16_[spec]`` | Specialization function*; 16bit   |
   |                                     | FX tensors                        |
   +-------------------------------------+-----------------------------------+
   ||                          *HWC Data Layout*                             |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_hwc_fx8``         | General function; 8bit FX         |
   |                                     | elements;                         |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_avepool_hwc_fx16``        | General function; 16bit FX        |
   |                                     | elements;                         |
   +-------------------------------------+-----------------------------------+


.. attention::
   \*For specialization          
   functions, backward          
   compatibility between        
   different releases cannot be  
   guaranteed. The General       
   functions call the available  
   specializations when possible.

.. _conditions-for-applying-the-function-3:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that you satisfy the following conditions before applying the
function:

-  Input tensor must be valid (see :ref:`mli_tns_struct`) and its rank
   must be 3.

-  Before processing, the output tensor must contain a valid pointer to
   a buffer with sufficient capacity (enough for result storing).

-  While processing, the following output tensor parameters are filled
   by functions:

   -  Shape (new shape is calculated according to input tensor shape,
      stride and padding parameters).

   -  Rank, element type, and element parameters (are copied from the input
      tensor).

   -  ``padding_top`` and ``padding_bottom`` parameters must be in range of [0,
      kernel_height).

   -  ``padding_left`` and ``padding_right`` parameters must be in range of [0,
      kernel_width).

   -  ``stride_width`` and ``stride_height`` parameters must be >= 1.
 
   -  ``kernel_width`` and ``kernel_height`` must be less than or equal to the
      corresponding dimensions of input tensor.

-  Additional restrictions for specialized functions are described in
   section :ref:`spec_fns`.