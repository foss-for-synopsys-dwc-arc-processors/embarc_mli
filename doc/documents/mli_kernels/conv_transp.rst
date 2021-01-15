Transpose Convolution Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel implements a general 2D transposed convolution operation 
which work by swapping the forward and backward passes of a convolution. 
For more details on calculations see chapter 4 of [2]

Optionally, saturating ReLU activation function can be applied to the 
result of the convolution during the function’s execution. For more info 
on supported ReLU types and calculations see :ref:`relu_prot`.

Dilation parameter of convolutions config isn’t applicable in MLI transposed 
convolution and is ignored.

Kernels which implement a Transpose Convolutions have the following prototype:

.. code::

   mli_status mli_krn_transpose_conv2d_hwcn_<data_format>(
      const mli_tensor *in,
      const mli_tensor *weights,
      const mli_tensor *bias,
      const mli_conv2d_cfg *cfg,
      mli_tensor *out);
..

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` 
and the function parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +---------------+-----------------------+------------------------------------------------------------------------+
   | **Parameter** | **Type**              | **Description**                                                        |
   +===============+=======================+========================================================================+
   | ``in``        | ``mli_tensor *``      | [IN] Pointer to constant input tensor.                                 |
   +---------------+-----------------------+------------------------------------------------------------------------+
   | ``weights``   | ``mli_tensor *``      | [IN] Pointer to constant weights tensor.                               |
   +---------------+-----------------------+------------------------------------------------------------------------+
   | ``bias``      | ``mli_tensor *``      | [IN] Pointer to constant bias tensor.                                  |
   +---------------+-----------------------+------------------------------------------------------------------------+
   | ``cfg``       | ``mli_conv2d_cfg *``  | [IN] Pointer to convolution parameters structure.                      |
   +---------------+-----------------------+------------------------------------------------------------------------+
   | ``out``       | ``mli_tensor *``      | [OUT] Pointer to output feature map tensor. Result is stored here.     |
   +---------------+-----------------------+------------------------------------------------------------------------+
..

The following table lists all the available Transpose Convolution functions:

.. table:: List of Available Transpose Convolution Functions
   :align: center
   :widths: auto 
   
   +------------------------------------------------------+--------------------------------------+
   | Function Name                                        | Details                              |
   +======================================================+======================================+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32``       || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out/weights data format: **sa8** |
   |                                                      || Bias data format: **sa32**          |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16``               || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || All tensors data format: **fx16**   |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8``       || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out data format: **fx16**        |
   |                                                      || Wights/Bias data format: **fx8**    |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k3x3``  || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out/weights data format: **sa8** |
   |                                                      || Bias data format: **sa32**          |
   |                                                      || Width of weights tensor: **3**      |
   |                                                      || Height of weights tensor: **3**     |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_k3x3``          || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || All tensors data format: **fx16**   |
   |                                                      || Width of weights tensor: **3**      |
   |                                                      || Height of weights tensor: **3**     |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k3x3``  || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out data format: **fx16**        |
   |                                                      || Wights/Bias data format: **fx8**    |
   |                                                      || Width of weights tensor: **3**      |
   |                                                      || Height of weights tensor: **3**     |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k5x5``  || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out/weights data format: **sa8** |
   |                                                      || Bias data format: **sa32**          |
   |                                                      || Width of weights tensor: **5**      |
   |                                                      || Height of weights tensor: **5**     |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_k5x5``          || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || All tensors data format: **fx16**   |
   |                                                      || Width of weights tensor: **5**      |
   |                                                      || Height of weights tensor: **5**     |
   +------------------------------------------------------+--------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k5x5``  || In/out layout: **HWC**              |
   |                                                      || Weights layout: **HWCN**            |
   |                                                      || In/out data format: **fx16**        |
   |                                                      || Wights/Bias data format: **fx8**    |
   |                                                      || Width of weights tensor: **5**      |
   |                                                      || Height of weights tensor: **5**     |
   +------------------------------------------------------+--------------------------------------+
..

All the listed functions must comply to the following conditions: 

 - ``in``, ``weights`` and ``bias`` tensors must be valid.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity and 
   valid ``el_params`` union. Other fields of the structure do not have to contain valid 
   data and are filled by the function.
	
 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``Mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - Channel (C) dimension of ``in`` and ``weights`` tensors must be equal.
 
 - ``bias`` must be a one-dimensional tensor. Its length must be equal to N dimension 
   (number of filters) of ``weights`` tensor.
   
 - ``padding_top`` and ``padding_bottom`` parameters must be in range of [(0, weights (H)eight).
 
 - ``padding_left`` and ``padding_right`` parameters must be in range of [(0, weights (W)idth).
 
 - ``stride_width`` and ``stride_height`` parameters must not be equal to 0.
 
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions:

 - ``in`` and out ``tensor`` must be quantized on the tensor level. It implies that each tensor 
   contains a single scale factor and a single zero offset.
   
 - ``weights`` and ``bias`` tensors must be symmetric. Both of them must be quantized on the same level. 
   Allowed Options:
   
   - Per Tensor level. It implies that each tensor contains a single scale factor and a single 
     zero offset equal to 0.
	 
   - Per N dimension level (number of filters). It implies that each tensor contains separate 
     scale point for each sub-tensor. All tensors contain single zero offset equal to 0.
	 
Scale factors of bias tensor must be equal to the multiplication of input scale factor broadcasted 
on weights array of scale factors. 

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.
