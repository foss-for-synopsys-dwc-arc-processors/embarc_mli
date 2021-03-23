.. _chap_element_wise:

Element-wise Group
------------------

The Element-wise Group describes operations that are applied element-by-element 
on two tensors of the same shape and return a tensor of the same shape. These kernels 
can also be used for broadcasting a scalar value. One of the input tensors can be 
a scalar tensor. In that case the operation is applied to the scalar value and each 
element of the other tensor.
 
:math:`\text{out}_{i} = operation(\text{in}_{i}^{1},\ \text{in}_{i}^{2}`)

:math:`\text{out}_{i} = operation(\text{in}_{\text{scalar}}^{1},\ \text{in}_{i}^{2}`)

Kernels which implement Element-wise functions have the following prototype:

.. code:: c

   mli_status mli_krn_eltwise_<operation>_<datatype> (
      const mli_tensor *in1,
      const mli_tensor *in2,
      mli_tensor *out);
..

.. _t_elw_data_conv:
.. table:: Element-wise Group Function Parameters
   :align: center
   :widths: auto 
   
   +---------------+-------------------+----------------------------------------------------------+
   | **Parameter** | **Type**          | **Description**                                          |
   +===============+===================+==========================================================+
   | ``in1``       | ``mli_tensor *``  | [IN] Pointer to constant input tensor.                   |
   +---------------+-------------------+----------------------------------------------------------+
   | ``in2``       | ``mli_tensor *``  | [IN] Pointer to constant input tensor.                   |
   +---------------+-------------------+----------------------------------------------------------+
   | ``out``       | ``mli_tensor *``  | [OUT] Pointer to output tensor. Result is stored here,   |
   |               |                   | capacity of this tensor should be large enough to store  |
   |               |                   | the results.                                             |
   +---------------+-------------------+----------------------------------------------------------+   
..

.. table:: List of Available Element-Wise Functions
   :align: center
   :widths: auto 
   
   +--------------------------------+---------------+---------------------------------+
   | **Function Name**              | **Operation** | **in1 / in2 / out data format** |
   +================================+===============+=================================+
   | ``mli_krn_eltwise_add_sa8``    | Addition      | **sa8**                         |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_add_fx16``   | Addition      | **fx16**                        |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_sub_sa8``    | Subtract      | **sa8**                         |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_sub_fx16``   | Subtract      | **fx16**                        |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_min_sa8``    | Minimum       | **sa8**                         |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_min_fx16``   | Minimum       | **fx16**                        |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_max_sa8``    | Maximum       | **sa8**                         |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_max_fx16``   | Maximum       | **fx16**                        |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_mul_sa8``    | Multiply      | **sa8**                         |
   +--------------------------------+---------------+---------------------------------+
   | ``mli_krn_eltwise_mul_fx16``   | Multiply      | **fx16**                        |
   +--------------------------------+---------------+---------------------------------+   
..

Ensure that you satisfy the following conditions before calling the listed functions:

 - ``in1`` and ``in2`` tensors must be valid and must share the same ``el_type`` field value. 
   They must be of the same shape, or one of them can be a tensor-scalar (see data field description 
   in the Table :ref:`mli_tnsr_struc`) 

 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity and valid ``el_params`` union. 
   Other fields of the structure do not have to contain valid data and are filled by the function

    - ``shape`` (a new shape is calculated according to input tensor shape, stride, and padding parameters).

    - ``rank``, ``el_type`` (these are copied from the input tensor).

 - For ``mli_krn_eltwise_min_*`` and ``mli_krn_eltwise_max_*`` functions, 
   the following additional restrictions apply

   - ``in1`` and ``in2`` must have the same quantization parameters. It means that ``el_params``
     union of tensors must be the same. For other elementwise functions this restriction is not applicable.

   - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity. 
     Other fields of the structure do not have to contain valid data and are filled by the function 
     (``shape``, ``rank``, ``el_params``, etc). 
 
 - If the result of an operation is out of container’s range, it is saturated to the 
   container’s limit.
   
 - The kernel supports in-place computation. It means that output and input tensor structures 
   can point to the same memory with the same memory strides but without shift.
   It can affect performance for some platforms.
   
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

 - For sa8 input and output tensors must be quantized on the tensor level. This implies 
   that each tensor contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

