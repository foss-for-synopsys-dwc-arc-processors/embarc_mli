.. _grp_conv:

Group Convolution Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

This kernel implements a grouped 2D convolution which applies general 2D 
convolution operation on a separate subset (groups) of inputs. In grouped 
convolutions with :math:`M` number of groups, the input and kernel are split by 
their channels to form :math:`M` distinct groups. Each group performs convolutions 
independent of the other groups to give :math:`M` different outputs. These individual 
outputs are then concatenated together to give the final output.  

For example, in a HWCN data layout, if the ``in`` feature map is :math:`(Hi, Wi, Ci)` and 
the ``weights`` is :math:`(Hk, Wk, Cw, Co)`, the ``output`` feature map is :math:`(Ho, Wo, Co)`
tensor where :math:`Ci` is equal to :math:`Cw * M` and :math:`Co` is multiple of :math:`M`.
Also spatial dimensions :math:`H*, W*` comply with the system of equations :eq:`eq_conv2d_shapes`. 

Depthwise convolution (see :ref:`conv_depthwise`) is an extreme case of group 
convolution with number of groups :math:`M` equal to number of input channels :math:`Ci`, and 
with the single filter per each group. TensorFlow-like “channel multiplier” 
functionality of depthwise convolution can be expressed by group convolution 
with number of groups equal to input channels :math:`Ci` and output channels :math:`Co` equal to 
channel multiplier number of filters per each group. 


.. note::

   For more details on group convolutions, see `ImageNet classification with deep 
   convolutional neural networks <https://dl.acm.org/doi/10.1145/3065386>`_ and 
   `Aggregated Residual Transformations for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
..

Optionally, saturating ReLU activation function can be applied to the result of 
the convolution during the function's execution. For more information on supported ReLU 
types and calculations, see :ref:`relu_prot`.

This is a MAC-based kernel which implies accumulation. See :ref:`quant_accum_infl` for more information on related quantization aspects. 
The Number of accumulation series is equal to :math:`(Hk * Wk * Cw)`.

Functions
^^^^^^^^^

Kernels which implement a group convolution have the following prototype:

.. code:: c

   mli_status mli_krn_group_conv2d_hwcn_<data_format>(
      const mli_tensor *in,
      const mli_tensor *weights,
      const mli_tensor *bias,
      const mli_conv2d_cfg *cfg,
      mli_tensor *out);
..
	  
where ``data_type`` is one of the data types listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Group Convolution Function Parameters
   :align: center
   :widths: auto 
   
   +---------------+------------------------+---------------------------------------------------+
   | **Parameter** | **Type**               | **Description**                                   |
   +===============+========================+===================================================+
   | ``in``        | ``mli_tensor *``       | [IN] Pointer to constant input tensor.            |
   +---------------+------------------------+---------------------------------------------------+
   | ``weights``   | ``mli_tensor *``       | [IN] Pointer to constant weights tensor.          |
   +---------------+------------------------+---------------------------------------------------+
   | ``bias``      | ``mli_tensor *``       | [IN] Pointer to constant bias tensor.             |
   +---------------+------------------------+---------------------------------------------------+
   | ``cfg``       | ``mli_conv2d_cfg *``   | [IN] Pointer to convolution parameters structure. |
   +---------------+------------------------+---------------------------------------------------+
   | ``out``       | ``mli_tensor *``       | [IN | OUT] Pointer to output feature map tensor.  |
   |               |                        | Result is stored here                             |
   +---------------+------------------------+---------------------------------------------------+
..

Number of groups to split is not provided to the kernel explicitly. Instead, it 
is derived from input and weights tensors shape. For example, in a HWCN data 
layout, if the ``in`` feature map is :math:`(Hi, Wi, Ci)` and the ``weights`` 
tensor is :math:`(Hk, Wk, Cw, Co)`, number of groups is :math:`M = Ci / Cw`, and 
number of filters per each group is :math:`Co / M`. 
Therefore, number of input channels :math:`Ci` must be a multiple of :math:`Cw`, and number of 
output channels :math:`Co` must be a multiple of number of groups :math:`M`. 

Here is a list of all available Group Convolution functions:

.. table:: List of Available Group Convolution Functions
   :align: center
   :widths: auto 

   +--------------------------------------------------+--------------------------------------+
   | Function Name                                    | Details                              |
   +==================================================+======================================+
   | ``mli_krn_group_conv2d_hwcn_sa8_sa8_sa32``       || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out/weights data type: **sa8**   |
   |                                                  || Bias data type: **sa32**            |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16``               || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || All tensors data type: **fx16**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16_fx8_fx8``       || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out data format: **fx16**        |
   |                                                  || Weights/Bias data format: **fx8**   |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16_k3x3``          || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || All tensors data format: **fx16**   |
   |                                                  || Width of weights tensor: **3**      |
   |                                                  || Height of weights tensor: **3**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k3x3``  || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out/weights data format: **sa8** |
   |                                                  || Bias data format: **sa32**          |
   |                                                  || Width of weights tensor: **3**      |
   |                                                  || Height of weights tensor: **3**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16_fx8_fx8_k3x3``  || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out data format: **fx16**        |
   |                                                  || Weights/Bias data format: **fx8**   |
   |                                                  || Width of weights tensor: **3**      |
   |                                                  || Height of weights tensor: **3**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_sa8_sa8_sa32_k5x5``  || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out/weights data format: **sa8** |
   |                                                  || Bias data format: **sa32**          |
   |                                                  || Width of weights tensor: **5**      |
   |                                                  || Height of weights tensor: **5**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16_k5x5``          || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || All tensors data format: **fx16**   |
   |                                                  || Width of weights tensor: **5**      |
   |                                                  || Height of weights tensor: **5**     |
   +--------------------------------------------------+--------------------------------------+
   | ``mli_krn_group_conv2d_hwcn_fx16_fx8_fx8_k5x5``  || In/out layout: **HWC**              |
   |                                                  || Weights layout: **HWCN**            |
   |                                                  || In/out data format: **fx16**        |
   |                                                  || Weights/Bias data format: **fx8**   |
   |                                                  || Width of weights tensor: **5**      |
   |                                                  || Height of weights tensor: **5**     |
   +--------------------------------------------------+--------------------------------------+
..

Conditions
^^^^^^^^^^
Ensure that you satisfy the following general conditions before calling the function:


- ``in``, ``out``, ``weights`` and ``bias`` tensors must be valid (see :ref:`mli_tnsr_struc`)
  and satisfy data requirements of the selected version of the kernel.

-  Shapes of ``in``, ``out``, ``weights`` and ``bias`` tensors must be compatible,
   which implies the following requirements:

    - ``in`` and ``out`` are 3-dimensional tensors (rank==3). Dimensions meaning, 
      and order (layout) is aligned with the specifc version of kernel.
 
    - ``weights`` is a 4-dimensional tensor (rank==4). Dimensions meaning, 
      and order (layout) is aligned with the specific kernel.
 
    - ``bias`` must be a one-dimensional tensor (rank==1). Its length must be equal to 
      :math:`Co` (output channels OR number of filters).
 
    - Channel :math:`Ci` dimension of ``in`` tensor must be multiple :math:`Cw` channel dimension 
      of ``weights`` tensors (:math:`Ci = M * Cw`).
 
    - :math:`Co` of ``weights`` tensor (output channels OR number of filters) must be multiple of number of 
      groups e.g. :math:`Co = M * X` where :math:`X` is the number of filters per group.
 
    - Shapes of ``in``, ``out`` and ``weights`` tensors together with ``cfg`` structure 
      must satisfy the equations :eq:`eq_conv2d_shapes`
 
    - Effective width and height of the ``weights`` tensor after applying dilation factor 
      (see :eq:`eq_conv2d_shapes`) must not exceed appropriate dimensions of the ``in`` tensor. 

 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``padding_top`` and ``padding_bottom`` parameters must be in the range of [0, :math:`\hat{Hk}`)
   where :math:`\hat{Hk}` is the effective kernel height (See :eq:`eq_conv2d_shapes`)
 
 - ``padding_left`` and ``padding_right`` parameters must be in the range of [0, :math:`\hat{Wk}`)
   where :math:`\hat{Wk}` is the effective kernel width (See :eq:`eq_conv2d_shapes`)
 
 - ``stride_width`` and ``stride_height`` parameters must not be equal to 0.

 - ``dilation_width`` and ``dilation_height`` parameters must not be equal to 0.

For **fx16** and **fx16_fx8_fx8** versions of kernel, in addition to the general conditions, ensure that you 
satisfy the following quantization conditions before calling the function:

 - The number of ``frac_bits`` in the ``bias`` and ``out`` tensors must not exceed the sum of ``frac_bits`` 
   in the ``in`` and ``weights`` tensors.

For **sa8_sa8_sa32** versions of kernel, in addition to the general conditions, ensure that you 
satisfy the following quantization conditions before calling the function:

 - ``in`` and ``out`` tensor must be quantized on the tensor level. This implies that each tensor 
   contains a single scale factor and a single zero offset.
   
 - Zero offset of ``in`` and ``out`` tensors must be within [-128, 127] range.
 
 - ``weights`` and ``bias`` tensors must be symmetric. Both must be quantized on the same level. 
   Allowed Options:
   
   - Per Tensor level. This implies that each tensor contains a single scale factor and a single 
     zero offset equal to 0.
	 
   - Per :math:`Co` dimension level (number of filters). This implies that each tensor contains separate 
     scale point for each sub-tensor. All tensors contain single zero offset equal to 0.
	 
 - Scale factors of bias tensor must be equal to the multiplication of input scale factor 
   broadcasted on weights array of scale factors.  See the example for the similar condition 
   in the :ref:`conv_2d`.

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the other fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

