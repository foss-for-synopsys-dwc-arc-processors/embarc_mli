.. _elmwise_mult:

Element-wise Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This kernel multiplies two tensors of the same shape element-wise and
   store results to the output tensor saving shape of inputs:

.. math:: y_{i} = {x1}_{i}*{x2}_{i}

..

   It supports simple broadcasting of single value (scalar tensor see 
   :ref:`mli_tns_struct`) on gen-eral tensor also. One of the operands 
   can be a scalar:

  It supports simple broadcasting of single value (scalar tensor see :ref:`mli_tns_struct`) on general tensor also. One of the operands can be a scalar:

.. math:: y_{i} = x_{i}*x\_\text{scalar}

..

   Elements of Input tensors must be of the same type but element
   parameter might differ. Output tensor must provide information about
   output format (element parameter).

   Output tensor holds the same element type, shape and rank parameters
   as the input tensors (for example, a scalar tensor).

   The kernel can perform in-place computation (output and input can
   point to exactly the same memory but without shift). It can affect
   performance for some platforms.

.. _function-configuration-structure-15:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for the kernel is required. All necessary
   information is provided by tensors.

.. _api-11:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_eltwise_mul_fx (           |
|                       |    const mli_tensor *in1,                     |
|                       |    const mli_tensor *in2,                     |
|                       |    mli_tensor *out);                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in1``               | [IN] Pointer to the   |
|                       |                       | first input tensor    |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``in2``               | [IN] Pointer to the   |
|                       |                       | second input tensor   |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-11:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+------------------------------+--------------------------------------+
| **Function**                 | **Description**                      |
+==============================+======================================+
| ``mli_krn_eltwise_mul_fx8``  | General function; 8bit FX elements;  |
+------------------------------+--------------------------------------+
| ``mli_krn_eltwise_mul_fx16`` | General function; 16bit FX elements; |
+------------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-11:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`)

   -  One of the inputs might be a valid scalar (see :ref:`mli_tns_struct`)

   -  If both of the input tensors are tensors, the shape and rank of both
      of them must be equal

   -  Before processing, the output tensor does not have to contain valid
      shape, rank and element type - they are filled by the function.

   -  Before processing, output tensor must contain a valid pointer to a
      buffer with sufficient capacity enough for storing the result. It
      also must contain valid element parameter (``el_params.fx.frac_bits``)