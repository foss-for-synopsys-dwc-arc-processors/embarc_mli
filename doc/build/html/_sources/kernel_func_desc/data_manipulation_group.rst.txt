Data Manipulation Group
-----------------------

   Data manipulation group provides operations to perform tensors
   element reordering, complex copying, and so on. Typically data manipulation
   does not change values itself.

.. _concat:
   
Concatenation
~~~~~~~~~~~~~

   This kernel concatenates multiple input tensors along one dimension
   to produce a single output tensor.
  
   The kernel takes an array of pointers to a set of input tensors. 
   The Kernel configuration structure contains the number of input tensors 
   (number of pointer in the array) and the axis along which 
   concatenation should be performed. The shape of all input tensors must
   be the same except the target dimension for concatenation.

   For example, concatenating tensors of shape [2, 4, 8] with tensor of
   shape [2, 6, 8] along axis=1 produces a tensor of shape [2, 10, 8],
   where the second dimension is the sum of the respective dimensions in
   input tensors. For the same example tensors, concatenation along
   other dimensions (axis=0 or axis=2) cannot be performed due to
   unequal values of the second dimension.

   Axis value must reflect index of shape in the array, which means that
   for the first dimension, axis=0; for the second, axis=1, and so on. Axis
   can not be a negative number.

   The number of tensors for concatenation is limited, but can be configured
   by changing ``MLI_CONCAT_MAX_TENSORS`` define (default: 8) in the
   ``MLI_config.h`` header file. It can slightly affect stack memory
   requirements of the kernel.

.. note::
   Ensure that input and output   
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.

.. _function-configuration-structure-16:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Definition**        | typedef struct {                              |
|                       | uint8_t tensors_num;                          |
|                       | uint8_t axis;                                 |
|                       | } mli_concat_cfg;                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Fields**            | ``tensors_num``       | Number of tensors to  |
|                       |                       | concatenate (number   |
|                       |                       | of pointers in        |
|                       |                       | “inputs” array)       |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``axis``              | Axis for              |
|                       |                       | concatenation         |
|                       |                       | (dimension number     |
|                       |                       | starting from 0)      |
+-----------------------+-----------------------+-----------------------+

.. _api-12:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_concat_fx (                |
|                       | const mli_tensor **inputs,                    |
|                       | const mli_concat_cfg *cfg,                    |
|                       | mli_tensor *out);                             |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``inputs``            | [IN] Pointer to the   |
|                       |                       | array of pointers to  |
|                       |                       | tensors for           |
|                       |                       | concatenation         |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | configuration         |
|                       |                       | structure with axis   |
|                       |                       | and amount of inputs  |
|                       |                       | definition            |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to the  |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-12:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-------------------------+--------------------------------------+
| **Function**            | **Description**                      |
+=========================+======================================+
| ``mli_krn_concat_fx8``  | General function; 8bit FX elements;  |
+-------------------------+--------------------------------------+
| ``mli_krn_concat_fx16`` | General function; 16bit FX elements; |
+-------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-12:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`)

   -  Element types and element parameters of all input tensors must be the
      same

   -  Rank of all input tensors must be the same.

   -  The shape of all input tensors must be the same except the dimension
      along which concatenation is performed (defined by axis field of
      kernel configuration)

   -  Number of input tensor must be less or equal to
      ``MLI_CONCAT_MAX_TENSORS`` value (defined in the the ``MLI_config.h``
      header file)

   -  Before processing, the output tensor must contain valid pointer to a
      buffer, with sufficient capacity enough for storing the result
      (that is, the total amount of elements of all input tensors).
      Other fields are filled by the kernel (shape, rank and element
      specific parameters)

   -  Buffers of all input and output tensors must point to different
      not-overlapped memory regions.

.. _permute:
     
Permute
~~~~~~~

   The kernel permutes dimensions of input tensor according to provided
   order. In other words, it transposes input tensors.

   The new order of dimensions is given by ``perm_dim`` array of kernel
   configuration structure. Output dimension ``#idx`` corresponds to the
   dimension of input tensor with ``#perm_dim[idx]``. Tensor’s data is
   reordered according to new shape.

   For example, if input tensors have the shape [2, 4, 8] and ``perm_dim``
   order is (2, 0, 1) then output tensor is of the shape [8, 2, 4]. This
   transpose reflects changing the feature map layout from HWC to CHW.

.. note::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.

.. _function-configuration-structure-17:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Definition**        | typedef struct {                              |
|                       | int perm_dim[MLI_MAX_RANK];                   |
|                       | } mli_permute_cfg;                            |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Fields**            | ``perm_dim``          | A permutation array.  |
|                       |                       | Dimensions order for  |
|                       |                       | output tensor.        |
+-----------------------+-----------------------+-----------------------+

.. _api-13:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_permute_fx (               |
|                       | const mli_tensor *in,                         |
|                       | const mli_permute_cfg *cfg,                   |
|                       | mli_tensor *out);                             |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | configuration         |
|                       |                       | structure with        |
|                       |                       | permutation order     |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to the  |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-13:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+--------------------------+--------------------------------------+
| **Function**             | **Description**                      |
+==========================+======================================+
| ``mli_krn_permute_fx8``  | General function; 8bit FX elements;  |
+--------------------------+--------------------------------------+
| ``mli_krn_permute_fx16`` | General function; 16bit FX elements; |
+--------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-13:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`)

   -  Only first N=rank_of_input_tensor values in permutation order array
      are considered by kernel. All of them must be unique, nonnegative
      and less then rank of the input tensor

   -  Before processing, output tensor must contain a valid pointer to a
      buffer with sufficient capacity enough for storing the result
      (that is, the total amount of elements in input tensor). Other
      fields are filled by kernel (shape, rank and element specific
      parameters)

   -  Buffers of input and output tensors must point to different
      not-overlapped memory regions.

.. _pad_2d:

Padding 2D
~~~~~~~~~~

   This kernel performs zero padding of borders across height and width
   dimensions of vision-specific input feature maps (see :ref:`fns`).

   Padding for each side of image (top, bottom, left, right) is
   configured separately according to input configuration structure, but
   the same padding for each side is used across all channels. Padding
   for HWC and CHW layouts of input tensor is implemented as separate
   functions. Output is calculated in the following order:

.. math:: out_channels = in_channels 

.. math:: out\_ height = in\_ height\  + padding\_ top + padding\_ bottom

.. math:: out\_ width = in\_ width\  + padding\_ left + padding\_ right

..

   For example, input tensor of the shape [2, 4, 8] representing image
   of CHW layout (channel dimension is the first value, so the image
   consists of 2 channels, 4 rows and 8 columns each). After applying
   the padding on the top and the right, with the ``padding_top=2`` and
   ``padding right=1``, an output image of shape [2, 6, 9] is produced,
   where top 2 rows contains only zeros, and last value of each row
   also equal to zero.

.. note::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.                

.. _function-configuration-structure-18:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Definition**        | typedef struct {                              |
|                       | uint8_t padding_left;                         |
|                       | uint8_t padding_right;                        |
|                       | uint8_t padding_top;                          |
|                       | uint8_t padding_bottom;                       |
|                       | } mli_padding2d_cfg;                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Fields**            | ``padding_left``      | Number of zero points |
|                       |                       | added to the left     |
|                       |                       | side of input (width  |
|                       |                       | dimension).           |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``padding_right``     | Number of zero points |
|                       |                       | added to the right    |
|                       |                       | side of input (width  |
|                       |                       | dimension).           |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``padding_top``       | Number of zero points |
|                       |                       | added to the upper    |
|                       |                       | side of input (height |
|                       |                       | dimension).           |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``padding_bottom``    | Number of zero points |
|                       |                       | added to the bottom   |
|                       |                       | side of input (height |
|                       |                       | dimension).           |
+-----------------------+-----------------------+-----------------------+

.. _api-14:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_permute_fx (               |
|                       | const mli_tensor *in,                         |
|                       | const mli_padding2d_cfg *cfg,                 |
|                       | mli_tensor *out);                             |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``cfg``               | [IN] Pointer to       |
|                       |                       | configuration         |
|                       |                       | structure with        |
|                       |                       | padding parameters    |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to the  |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-14:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+--------------------------------+--------------------------------------+
| **Function**                   | **Description**                      |
+================================+======================================+
| *HWC Data Layout*              |                                      |
+--------------------------------+--------------------------------------+
| ``mli_krn_padding2d_hwc_fx8``  | General function; 8bit FX elements;  |
+--------------------------------+--------------------------------------+
| ``mli_krn_padding2d_hwc_fx16`` | General function; 16bit FX elements; |
+--------------------------------+--------------------------------------+
| *СHW Data Layout*              |                                      |
+--------------------------------+--------------------------------------+
| ``mli_krn_padding2d_сhw_fx8``  | General function; 8bit FX elements;  |
+--------------------------------+--------------------------------------+
| ``mli_krn_padding2d_сhw_fx16`` | General function; 16bit FX elements; |
+--------------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-14:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`)

   -  Before processing, output tensor must contain a valid pointer to a
      buffer with sufficient capacity enough for storing the result
      (that is, the total amount of elements in input tensor). Other
      fields are filled by kernel (shape, rank and element specific
      parameters)

   -  Buffers of input and output tensors must point to different
      not-overlapped memory regions

