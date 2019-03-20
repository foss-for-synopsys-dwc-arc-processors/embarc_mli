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
|                       |    uint8_t tensors_num;                       |
|                       |    uint8_t axis;                              |
|                       |  } mli_concat_cfg;                            |
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
|                       |    const mli_tensor **inputs,                 |
|                       |    const mli_concat_cfg *cfg,                 |
|                       |    mli_tensor *out);                          |
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