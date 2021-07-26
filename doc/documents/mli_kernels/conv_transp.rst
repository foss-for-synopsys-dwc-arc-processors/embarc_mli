Transpose Convolution Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

This kernel implements a general 2D transposed convolution operation 
which works by swapping the forward and backward passes of a convolution. 
For more details on calculations, see chapter 4 of `A guide to convolution 
arithmetic for deep learning <https://arxiv.org/abs/1603.07285>`_.

Optionally, a saturating ReLU activation function can be applied to the 
result of the convolution during the function's execution. For more info 
on supported ReLU types and calculations, see :ref:`relu_prot`.

The ``dilation_height`` and ``dilation_width`` parameter of ``mli_conv2d_cfg`` 
configuration structure is not applicable in MLI transposed convolution and must be equal to 1.

For example, in a HWCN data layout, if the ``in`` feature map is :math:`(Hi, Wi, Ci)` and 
the ``weights`` is :math:`(Hk, Wk, Ci, Co)`, the ``output`` feature map is :math:`(Ho, Wo, Co)`
tensor where the spatial dimensions comply with the following system of equations: 

.. math::
  :label: eq_transp_conv2d_shapes

  \begin{cases}

  \hat{Ph} = ({Hk}-1)*2-padding\_top-padding\_bottom

  \hat{Pw} = ({Wk}-1)*2-padding\_left-padding\_right

  \hat{Wi} = ({Wi}-1)*stride\_width+1 

  \hat{Hi} = ({Hi}-1)*stride\_height+1 

  {Wo} = \hat{Wi}+\hat{Pw}-{Wk}+1

  {Ho} = \hat{Hi}+\hat{Ph}-{Hk}+1

  \end{cases}
..

Where:

  :math:`\hat{Wi}`, :math:`\hat{Hi}` *- effective* ``in`` *feature map width and height
  after applying* :math:`stride\_*` *to the original width* (:math:`Wi`) *and height* (:math:`Hi`).

  :math:`\hat{Pw}`, :math:`\hat{Ph}` *- transposed width and height paddings.* 

  :math:`Wo`, :math:`Ho` *-* ``out`` *feature map width and height.*
  
  :math:`Wk`, :math:`Hk` *-* ``weights`` * kernel width and height.*

This is a MAC-based kernel which implies accumulation. See :ref:`quant_accum_infl` for more information on related quantization aspects. 
The Number of accumulation series is up to (:math:`Wk*Hk*Ci`).

Functions
^^^^^^^^^

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
   
   +---------------+-----------------------+---------------------------------------------------+
   | **Parameter** | **Type**              | **Description**                                   |
   +===============+=======================+===================================================+
   | ``in``        | ``mli_tensor *``      | [IN] Pointer to constant input tensor.            |
   +---------------+-----------------------+---------------------------------------------------+
   | ``weights``   | ``mli_tensor *``      | [IN] Pointer to constant weights tensor.          |
   +---------------+-----------------------+---------------------------------------------------+
   | ``bias``      | ``mli_tensor *``      | [IN] Pointer to constant bias tensor.             |
   +---------------+-----------------------+---------------------------------------------------+
   | ``cfg``       | ``mli_conv2d_cfg *``  | [IN] Pointer to convolution parameters structure. |
   +---------------+-----------------------+---------------------------------------------------+
   | ``out``       | ``mli_tensor *``      | [IN | OUT] Pointer to output feature map tensor.  |
   |               |                       | Result is stored here                             |
   +---------------+-----------------------+---------------------------------------------------+
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

Conditions
^^^^^^^^^^

Ensure that you satisfy the following conditions before calling the function: 

 - ``in``, ``out``, ``weights`` and ``bias`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the used version of the kernel.

 - Shapes of ``in``, ``out``, ``weights`` and ``bias`` tensors must be compatible,
   which implies the following requirements:

    - ``in`` and ``out`` are 3-dimensional tensors (rank==3). Dimensions meaning, 
      and order (layout) is aligned with the used version of kernel.

    - ``weights`` is a 4-dimensional tensor (rank==4). Dimensions meaning, 
      and order (layout) is aligned with the used kernel.

    - ``bias`` must be a one-dimensional tensor (rank==1). Its length must be equal to 
      :math:`Co` (output channels OR number of filters).

    - Channel :math:`Ci` dimension of ``in`` and ``weights`` tensors must be equal.

    - Shapes of ``in``, ``out`` and ``weights`` tensors together with ``cfg`` structure 
      must satisfy the equations :eq:`eq_transp_conv2d_shapes`

    - Width and height (:math:`Wk, Hk`) of the ``weights`` tensor must not exceed 
      appropriate effective dimensions of the ``in`` tensor (see :eq:`eq_transp_conv2d_shapes`). 

 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``padding_top`` and ``padding_bottom`` parameters must be in range of [0, weights (H)eight).
 
 - ``padding_left`` and ``padding_right`` parameters must be in range of [0, weights (W)idth).
 
 - ``stride_width`` parameter must be in range of [1, weights (W)idth).

 - ``stride_height`` parameter must be in range of [1, weights (H)eight).

 - ``dilation_height`` and ``dilation_width`` must be equal to 1. 

 
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function:

 - ``in`` and ``out`` tensor must be quantized on the tensor level. This implies that each tensor 
   contains a single scale factor and a single zero offset.
   
 - Zero offset of ``in`` and ``out`` tensors must be within [-128, 127] range.
 
 - ``weights`` and ``bias`` tensors must be symmetric. Both must be quantized on the same level. 
   Allowed Options:
   
   - Per Tensor level. This implies that each tensor contains a single scale factor and a single 
     zero offset equal to 0.

   - Per :math:`Co` dimension level (number of filters). This implies that each tensor contains separate 
     scale point for each sub-tensor. All tensors contain single zero offset equal to 0.

 - Scale factors of bias tensor must be equal to the multiplication of input scale factor broadcasted 
   on weights array of scale factors. See the example for the similar condition in the :ref:`conv_2d`.

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the rest fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.
