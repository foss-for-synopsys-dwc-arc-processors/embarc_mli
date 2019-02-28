Convolution Group
-----------------

   Convolution group of functions convolve input features with a set of
   trained filters. Typically these are multi-dimensional tensors. 

.. _cnvl_2d:
   
Convolution 2D
~~~~~~~~~~~~~~

.. _f_cnvl_2d_op:
.. figure:: ../pic/images/image104.jpg

   2D-Convolution Operation

   This kernel implements a general 2D convolution operation. It applies
   each filter of weights tensor to each framed area of the size of
   input tensor

   Input and output feature maps are three-dimensional tensor even if one of
   dimensions is equal to 1. Input shape includes height, width, and
   depth (channels) according to used data layout (see :ref:`data_muldim`).

   Filter (also referred as kernel in other sources) consists of two
   separate tensors:

   -  **weights tensor**: Weights is 4-dimensional tensor. The first
      dimension of weights tensor is the output depth (might be considered as
      the number of filters). The rest dimensions includes height, width
      and kernel depth in order of particular layout. Ensure that the
      layout and depth dimensions of Input and weights tensors are same.

   -  **Biases tensor**: Biases is one-dimensional tensor of shape
      [output_depth].

..

   In processing, perception window for applying the filter (see :numref:`f_cnvl_2d_op`
   ) moves along dimensions according to stride parameters.

   To implicitly insert additional points to sides of feature map
   (considering only width and height dimensions), ensure that you set
   the padding parameters. Padding influences how feature map is divided
   into patches for applying kernels because values of padded points are
   always zero.

   For example, in a HWC data layout, if input fmap is [Hi, Wi, Ci] and
   kernel is [Co, Hk, Wk, Ci], the output fmap is [Ho, Wo, Co] matrix
   where the output dimensions Ho and Wo are calculated dynamically
   depending on convolution parameters (such as padding and stride)
   inputs shape.

   For more details on calculations see convolution part of entry [3] in :ref:`refs`.

   ReLU activation function might be applied to result of convolution. The
   following types of ReLU activations are supported (for more info see
   :ref:`relu`):

   -  RELU_NONE

   -  RELU_GEN

   -  RELU_1

   -  RELU_6

.. note::
   Ensure that input and output 
   tensors do not point to      
   overlapped memory regions,   
   otherwise the behavior is    
   undefined.                    

.. _fn_conf_struct:
   
Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+---------------------------+
|                       |                                                   |
|                       |.. code:: c                                        |
|                       |                                                   |
| **Definition**        | typedef struct {                                  |
|                       |    uint8_t stride_width;                          |
|                       |    uint8_t  stride_height;                        |
|                       |    uint8_t padding_left;                          |
|                       |    uint8_t padding_right;                         |
|                       |    uint8_t padding_top;                           |
|                       |    uint8_t  padding_bottom;                       |
|                       |    mli_relu_cfg relu;                             |
|                       | } mli_conv2d_cfg;                                 |
|                       |                                                   |
+-----------------------+-----------------------+---------------------------+
| **Fields**            | ``stride_width``      | Stride of filter across   |
|                       |                       | width dimension of input  |
+-----------------------+-----------------------+---------------------------+
|                       | ``stride_height``     | Stride (step) of filter   |
|                       |                       | across height dimension   |
|                       |                       | of input                  |
+-----------------------+-----------------------+---------------------------+
|                       | ``padding_left``      | Number of zero points     |
|                       |                       | implicitly added to the   |
|                       |                       | leftside of input (width  |
|                       |                       | dimension)                |
+-----------------------+-----------------------+---------------------------+
|                       | ``padding_right``     | Number of zero points     |
|                       |                       | implicitly added to       |
|                       |                       | the right side of input   |
|                       |                       | (width dimension).        |
+-----------------------+-----------------------+---------------------------+
|                       | ``padding_top``       | Number of zero points     |
|                       |                       | implicitly added to the   |
|                       |                       | upper side of input       |
|                       |                       | (height dimension).       |
+-----------------------+-----------------------+---------------------------+
|                       | ``padding_bottom``    | Number of zero points     |
|                       |                       | implicitly added to the   |
|                       |                       | bottom side of input      |
|                       |                       | (height dimension).       |
+-----------------------+-----------------------+---------------------------+
|                       | ``relu``              | Type of ReLU activation   |
|                       |                       | applied to output values  |
+-----------------------+-----------------------+---------------------------+

General API
^^^^^^^^^^^

   Interface of all specializations are the same and described as follows:
   
   \

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status  mli_krn_conv2d_<layout>_          |                        
|                       | <data_type>[_specialization](                 |           
|                       | const mli_tensor *in,                         |                 
|                       | const mli_tensor *weights,                    |
|                       | const mli_tensor *bias,                       |
|                       | const mli_conv2d_cfg *cfg,                    |
|                       | mli_tensor *out);                             |
|                       |                                               |   
+-----------------------+-----------------------+-----------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | feature map tensor    |
+-----------------------+-----------------------+-----------------------+
|                       | ``weights``           | [IN] Pointer to       |
|                       |                       | convolution filters   |
|                       |                       | weights tensor        |
+-----------------------+-----------------------+-----------------------+
|                       | ``bias``              | [IN] Pointer to       |
|                       |                       | convolution filters   |
|                       |                       | biases tensor         |
+-----------------------+-----------------------+-----------------------+
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | convolution           |
|                       |                       | parameters structure  |
+-----------------------+-----------------------+-----------------------+
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output feature map    |
|                       |                       | tensor. Result is     |
|                       |                       | stored here           |
+-----------------------+-----------------------+-----------------------+

Function Specializations
^^^^^^^^^^^^^^^^^^^^^^^^

   There are about 70 specializations for the primitive, assuming
   various combinations of inputs parameters. Convolution primitive
   follows naming convention for specializations (see :ref:`spec_fns`).
   The ``mli_krn_conv2d_spec_api.h`` header file contains declarations of  
   all specializations for the primitive.

\

+-------------------------------------+-----------------------------------+
| **Function**                        | **Description**                   |
+=====================================+===================================+
| *CHW Data Layout*                                                       |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx8``          | Switching function; 8bit FX       |
|                                     | tensors; Delegates calculations   |
|                                     | to suitable specialization or     |
|                                     | generic function.                 |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx16``         | Switching function; 16bit FX      |
|                                     | tensors;                          |
|                                     |                                   |
|                                     | Delegates calculations to         |
|                                     | suitable specialization or        |
|                                     | generic function.                 |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx8w16d``      | General function; FX tensors      |
|                                     | (8bit weights and biases, 16 bit  |
|                                     | input and output)                 |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx8_generic``  | General function; 8bit FX tensors |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx16_generic`` | General function; 16bit FX        |
|                                     | tensors                           |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx8_[spec]``   | Specialization function*; 8bit FX |
|                                     | tensors                           |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_chw_fx16_[spec]``  | Specialization function*; 16bit   |
|                                     | FX tensors                        |
+-------------------------------------+-----------------------------------+
| *HWC Data Layout*                   |                                   |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_hwc_fx8``          | General function; 8bit FX tensors |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_hwc_fx16``         | General function; 16bit FX        |
|                                     | tensors                           |
+-------------------------------------+-----------------------------------+
| ``mli_krn_conv2d_hwc_fx8w16d``      | General function; FX tensors      |
|                                     | (8bit weights and biases, 16 bit  |
|                                     | input and output)                 |
+-------------------------------------+-----------------------------------+

.. note::
   \*For specialization functions, backward compatibility between different releases cannot be guaranteed. The General functions call the available specializations when possible.   
   
Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, Weights and Bias tensors must be valid (see :ref:`mli_tns_struct`)

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer with sufficient capacity (enough for result storing). It
      also must contain valid element parameter (``el_params.fx.frac_bits``)

   -  Before processing, the output tensor does not have to contain valid
      shape, rank and element type fields. These are filled by the
      function.

   -  The data layout of Input and weights tensors must be the same. Their
      depth (channels) dimension must also be equal.

   -  Bias must be a one-dimensional tensor. Its length must be equal to the
      amount of filters (first dimension of weights tensor).

   -  ``padding_top`` and ``padding_bottom`` parameters must be in range of [0,
      weights_height).

   -  ``padding_left`` and ``padding_right`` parameters must be in range of [0,
      weights_width).

   -  ``stride_width`` and ``stride_height`` parameters must not be equal to 0.

   -  ``weights_width`` and ``weights_height`` must be less than or equal to the
      appropriate dimensions of the input tensor.

   -  Additional restrictions for specialized functions are described in
      :ref:`spec_fns`.

Depth-wise Convolution
~~~~~~~~~~~~~~~~~~~~~~

.. _f_depthwise:
.. figure:: ../pic/images/image107.jpg
   
   Depth-wise Convolution Operation

   This kernel implements a 2D depth-wise convolution operation applying
   each filter channel to each input channel separately. As a result,
   output image depth is the same as input image depth.

   MLI implementation of depth-wise convolution is compatible with caffe
   implementation of convolution layer with group parameter equal to
   number of input channels. In comparison with TensorFlow
   implementation (``tf.nn.depthwise_conv2d`` in python API), this
   implementation does not support channel multiplier feature. Therefore,
   the last dimension of weights tensor must be equal to 1.

   For the example for CHW data layout, if the input feature map is [Ci,
   Hi, Wi] and the kernel is [Ci, Hk, Wk,1], the output feature map is
   [Ci, Ho, Wo] matrix where the output dimensions Ho and Wo are
   calculated dynamically according to convolution parameters (such as
   padding and stride) in the same way as for general convolution 2D
   kernel (see :ref:`cnvl_2d`).

   ReLU activation function might be applied to result of convolution. The
   following types of ReLU activations are supported (for more info see
   :ref:`relu`):

   -  RELU_NONE

   -  RELU_GEN

   -  RELU_1

   -  RELU_6

.. note::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.
   
.. _function-configuration-structure-1:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Depth-wise convolution kernel shares configuration structure with
   general convolution 2D kernel. For more information see :ref:`fn_conf_struct`
   section of Convolution 2D function.

.. _general-api-1:

General API
^^^^^^^^^^^

   Interface of all specializations are the same and described in
   following table.
   
   \

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_depthwise_conv2d_          |
|                       | <layout>_<data_type>[_specialization](        |
|                       |    const mli_tensor *in,                      |
|                       |    const mli_tensor *weights,                 |
|                       |    const mli_tensor *bias,                    |
|                       |    const mli_conv2d_cfg *cfg,                 |
|                       |    mli_tensor *out);                          |
+-----------------------+-----------------------+-----------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | feature map tensor    |
+-----------------------+-----------------------+-----------------------+
|                       | ``weights``           | [IN] Pointer to       |
|                       |                       | convolution filters   |
|                       |                       | weights tensor        |
+-----------------------+-----------------------+-----------------------+
|                       | ``bias``              | [IN] Pointer to       |
|                       |                       | convolution filters   |
|                       |                       | biases tensor         |
+-----------------------+-----------------------+-----------------------+
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | convolution           |
|                       |                       | parameters structure  |
+-----------------------+-----------------------+-----------------------+
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output feature map    |
|                       |                       | tensor. Result is     |
|                       |                       | stored here           |
+-----------------------+-----------------------+-----------------------+

.. _function-specializations-1:

Function Specializations
^^^^^^^^^^^^^^^^^^^^^^^^

   There are about 70 specializations for the primitive assuming various
   combinations of inputs parameters. Depth-wise convolution primitive
   follows the naming convention for specializations (see :ref:`spec_fns`). 
   The header file ``mli_krn_depthwise_conv2d_spec_api.h`` contains declarations 
   of all specializations for the primitive.

   :numref:`Non_spec_func_Dw_cnvl` does not contain specialized functions. 
   No functions for HWC data layout have been implemented at the moment. 
   To use depth-wise convolution in this case, ensure that you change weights
   and inputs layout beforehand by permute primitive (see :ref:`permute`).

.. _Non_spec_func_Dw_cnvl:
.. table:: Non-Specialized Functions for Depth-Wise Convolution
   :widths: auto
   
   +-----------------------------------------------+-----------------------------------+
   | **Function**                                  | **Description**                   |
   +===============================================+===================================+
   | ``mli_krn_depthwise_conv2d_chw_fx8``          | Switching function (see           |
   |                                               | :ref:`fns`); 8bit FX tensors;     |
   |                                               | Delegates calculations to         |
   |                                               | suitable specialization or        |
   |                                               | generic function.                 |
   +-----------------------------------------------+-----------------------------------+
   | ``mli_krn_depthwise_conv2d_chw_fx16``         | Switching function (see           |
   |                                               | :ref:`fns`); 16bit FX tensors;    |
   |                                               |                                   |
   |                                               | Delegates calculations to         |
   |                                               | suitable specialization or        |
   |                                               | generic function.                 |
   +-----------------------------------------------+-----------------------------------+
   | ``mli_krn_depthwise_conv2d_chw_fx8w16d``      | General function; 8bit FX tensors |
   |                                               |                                   |
   +-----------------------------------------------+-----------------------------------+
   | ``mli_krn_depthwise_conv2d_chw_fx8_generic``  | General function; 16bit FX        |
   |                                               | tensors                           |
   +-----------------------------------------------+-----------------------------------+
   | ``mli_krn_depthwise_conv2d_chw_fx16_generic`` | Specialization function*; 8bit FX |
   |                                               | tensors                           |
   +-----------------------------------------------+-----------------------------------+
   | ``mli_krn_depthwise_conv2d_chw_fx16_[spec]``  | Specialization function*; 16bit   |
   |                                               | FX tensors                        |
   +-----------------------------------------------+-----------------------------------+

.. note:: 
   \*For specialization         
   functions, backward          
   compatibility between        
   different releases cannot be  
   guaranteed. The General       
   functions call the available  
   specializations when possible.

.. _conditions-for-applying-the-function-1:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, Weights and Bias tensors must be valid (see :ref:`mli_tns_struct`)

   -  Before processing, the output tensor must contain valid pointer to a
      buffer with sufficient capacity (enough for result storing). It
      also must contain valid element parameter (``el_params.fx.frac_bits``)

   -  Before processing, the output tensor does not have to contain valid
      shape, rank and element type fields. These are filled by function.

   -  Input and weights tensors data layout must be the same. Amount of
      weights channels must be 1.

   -  Amount of filters (first dimension of weights tensor) must be equal
      to number of input channels.

   -  Bias must be one-dimensional tensor. Its length must be equal to
      amount of filters (first dimension of weights tensor)

   -  padding_top and padding_bottom parameters must be in range of [0,
      weights_height).

   -  ``padding_left`` and ``padding_right`` parameters must be in range of [0,
      weights_width).

   -  ``stride_width`` and ``stride_height`` parameters must not be equal to 0.

   -  ``weights_width`` and ``weights_height`` must be less or equal to appropriate
      dimensions of input tensor.

   -  Additional restrictions for specialized functions are described in
      section :ref:`spec_fns`.
	  