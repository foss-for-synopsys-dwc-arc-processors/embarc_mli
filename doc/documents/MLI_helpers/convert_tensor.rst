..  _conv_tensor:

Convert Tensor
~~~~~~~~~~~~~~

   This function copies elements from input tensor to output with data
   conversion according to the output tensor type parameters.

   For example, the function can:

   -  convert data according to new element type: ``fx16`` to ``fx8`` and backward

   -  change data according to new data parameter: increase/decrease the
      number of fractional bits while keeping the same element type for
      FX data

..

   Conversion is performed using

   -  rounding when the number of significant bits increases.

   -  saturation when the number of significant bits decreases.

..

   This operation does not change tensor shape. It copies it from input
   to output.

   Kernel can perform in-place computation, but only for conversions
   without increasing data size, so that that it does not lead to
   undefined behavior. Therefore, output and input might point to exactly the
   same memory (but without shift) except ``fx8`` to ``fx16`` conversion.
   In-place computation might affect performance for some platforms.

.. _api-18:

API
^^^

+-----------------------+-----------------------+----------------------------------------------+
| **Prototype**         |.. code:: c                                                           |
|                       |                                                                      |
|                       | mli_status mli_hlp_convert_tensor(mli_tensor *in, mli_tensor *out);  |
|                       |                                                                      |
+-----------------------+-----------------------+----------------------------------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input                        |
|                       |                       | tensor                                       |
+-----------------------+-----------------------+----------------------------------------------+
|                       | ``start_dim``         | [OUT] Pointer to                             |
|                       |                       | output tensor                                |
+-----------------------+-----------------------+----------------------------------------------+
| **Returns**           | ``status code``       |                                              |
+-----------------------+-----------------------+----------------------------------------------+

.. _conditions-for-applying-the-function-7:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   -  Input must be a valid tensor (see :ref:`mli_tns_struct`).

   -  Before processing the output tensor must contain a valid pointer to a
      buffer with sufficient capacity enough for storing the result
      (that is, the total amount of elements in input tensor).

   -  The output tensor also must contain valid element type and its
      parameter (``el_params.fx.frac_bits``)

   -  Before processing, the output tensor does not have to contain valid
      shape and rank - they are copied from input tensor.

