Transpose Convolution Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel implements a general 2D transposed convolution operation 
which works by swapping the forward and backward passes of a convolution. 
For more details on calculations, see chapter 4 of `A guide to convolution 
arithmetic for deep learning <https://arxiv.org/abs/1603.07285>`_.

Optionally, a saturating ReLU activation function can be applied to the 
result of the convolution during the function’s execution. For more info 
on supported ReLU types and calculations, see :ref:`relu_prot`.

The ``dilation_height`` and ``dilation_width`` parameter of ``mli_conv2d_cfg`` 
configuration structure is not applicable in MLI transposed convolution and is 
ignored.

This is a MAC-based kernel which implies accumulation. See :ref:`quant_accum_infl` for more info on related quantization aspects. 
The Number of accumulation series is up to (kernel_height * kernel_width * in_channels).

Kernels which implement Transpose Convolutions have the following prototype:

.. code:: c

   mli_status mli_krn_transpose_conv2d_hwcn_<data_format>(
      const mli_tensor *in,
      const mli_tensor *weights,
      const mli_tensor *bias,
      const mli_conv2d_cfg *cfg,
      mli_tensor *out);
..

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` 
and the function parameters are shown in the following table:

.. table:: Transpose Convolution Function Parameters
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
   
   +-----------------------------------------------------------+-----------------------------------------+
   | Function Name                                             | Details                                 |
   +===========================================================+=========================================+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32``            || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out/weights data format: **sa8**    |
   |                                                           || Bias data format: **sa32**             |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16``                    || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || All tensors data format: **fx16**      |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8``            || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out data format: **fx16**           |
   |                                                           || Wights/Bias data format: **fx8**       |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k2x2_str2``  || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out/weights data format: **sa8**    |
   |                                                           || Bias data format: **sa32**             |
   |                                                           || Width of weights tensor: **2**         |
   |                                                           || Height of weights tensor: **2**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_k2x2_str2``          || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || All tensors data format: **fx16**      |
   |                                                           || Width of weights tensor: **2**         |
   |                                                           || Height of weights tensor: **2**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k2x2_str2``  || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out data format: **fx16**           |
   |                                                           || Wights/Bias data format: **fx8**       |
   |                                                           || Width of weights tensor: **2**         |
   |                                                           || Height of weights tensor: **2**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_sa8_sa8_sa32_k4x4_str2``  || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out/weights data format: **sa8**    |
   |                                                           || Bias data format: **sa32**             |
   |                                                           || Width of weights tensor: **4**         |
   |                                                           || Height of weights tensor: **4**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_k4x4_str2``          || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || All tensors data format: **fx16**      |
   |                                                           || Width of weights tensor: **4**         |
   |                                                           || Height of weights tensor: **4**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
   | ``mli_krn_transpose_conv2d_hwcn_fx16_fx8_fx8_k4x4_str2``  || In/out layout: **HWC**                 |
   |                                                           || Weights layout: **HWCN**               |
   |                                                           || In/out data format: **fx16**           |
   |                                                           || Wights/Bias data format: **fx8**       |
   |                                                           || Width of weights tensor: **4**         |
   |                                                           || Height of weights tensor: **4**        |
   |                                                           || Stride across Width dimension: **2**   |
   |                                                           || Stride across Hight dimension: **2**   |
   +-----------------------------------------------------------+-----------------------------------------+
..

Ensure that you satisfy the following conditions before calling the function: 

 - ``in``, ``weights`` and ``bias`` tensors must be valid (see :ref:`mli_tnsr_struc`).
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity, valid 
   ``mem_stride`` field  and 
   valid ``el_params`` union. Other fields of the structure do not have to contain valid 
   data and are filled by the function.
	
 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - Channel (C) dimension of ``in`` and ``weights`` tensors must be equal.
 
 - ``bias`` must be a one-dimensional tensor. Its length must be equal to N dimension 
   (number of filters) of ``weights`` tensor.
   
 - ``padding_top`` and ``padding_bottom`` parameters must be in range of [0, weights (H)eight).
 
 - ``padding_left`` and ``padding_right`` parameters must be in range of [0, weights (W)idth).
 
 - ``stride_width`` and ``stride_height`` parameters must not be equal to 0.
 
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function:

 - ``in`` and ``out`` tensor must be quantized on the tensor level. This implies that each tensor 
   contains a single scale factor and a single zero offset.
   
 - Zero offset of ``in`` and ``out`` tensors must be within [-128, 127] range.
 
 - ``weights`` and ``bias`` tensors must be symmetric. Both must be quantized on the same level. 
   Allowed Options:
   
   - Per Tensor level. This implies that each tensor contains a single scale factor and a single 
     zero offset equal to 0.

   - Per N dimension level (number of filters). This implies that each tensor contains separate 
     scale point for each sub-tensor. All tensors contain single zero offset equal to 0.

 - Scale factors of bias tensor must be equal to the multiplication of input scale factor broadcasted 
   on weights array of scale factors. 

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.
