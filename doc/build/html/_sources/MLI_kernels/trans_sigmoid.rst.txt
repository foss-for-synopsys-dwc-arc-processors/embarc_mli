.. _sigmoid:

Sigmoid
~~~~~~~

.. image:: ../images/image152.png 
   :align: center
   :alt: Logistic (Sigmoid) function

..
   
   This kernel performs sigmoid (also mentioned as logistic) activation
   function on input tensor element-wise and stores the result to the
   output tensor:

.. math::

   f(x) = \frac{1}{1 + e^{- x}}

..

   *x* - input value.
   
..

   The range of function is (0, 1), and kernel outputs completely
   fractional tensor of the same shape and type as input. For ``fx8`` type,
   the output holds 7 fractional bits, and 15 fractional bits for ``fx16``
   type. For this reason, the maximum representable value of SoftMax is
   equivalent to 0.9921875 for ``fx8`` output tensor, and to
   0.999969482421875 for ``fx16`` (not 1.0).

   Kernel can perform in-place computation: output and input can point
   to exactly the same memory (the same starting address).

.. _function-configuration-structure-9:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for sigmoid kernel is required. All
   necessary information is provided by tensors.

.. _api-5:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_sigm_<data_type>(          |
|                       |    const mli_tensor *in,                      |
|                       |    mli_tensor *out);                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor for     |
|                       |                       | storing the result    |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-5:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+--------------------------------------+
| **Function**          | **Description**                      |
+=======================+======================================+
| ``mli_krn_sigm_fx8``  | General function; 8bit FX elements;  |
+-----------------------+--------------------------------------+
| ``mli_krn_sigm_fx16`` | General function; 16bit FX elements; |
+-----------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-5:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, tensors must be valid (see :ref:`mli_tns_struct`).

   -  Before processing, the output tensor must contain a valid pointer to
      a buffer, with sufficient capacity enough for storing the result.
      Other fields are filled by kernel (shape, rank and element
      specific parameters)