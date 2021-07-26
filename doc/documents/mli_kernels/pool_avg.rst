Average Pooling Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

Average pooling computes each value of the output tensor as the average of all values 
in the related perception area of a single channel of the input tensor. The perception 
area size is defined as :math:`kernel\_width * kernel\_height` (see Figure 
:ref:`f_pool2x2avg`).

.. _f_pool2x2avg:  
.. figure::  ../images/pool_2x2avg.png
   :align: center

   Example for 2x2 Average pooling
..

For example, in a HWC data layout, if the ``in`` feature map is :math:`(Hi, Wi, Ci)`,
the ``out`` feature map is :math:`(Ho, Wo, Ci)` tensor where the spatial dimensions 
comply with the system of equations :eq:`eq_pool_shapes`. 

Functions
^^^^^^^^^

Kernels which implement average pooling functions have the following prototype:

.. code:: c

   mli_status  mli_krn_avepool_hwc_<data_format>(
      const mli_tensor *in,
      const mli_pool_cfg *cfg,
      mli_tensor *out);
..

.. table:: Average Pooling Function Parameters
   :align: center
   :widths: auto
   	  
   +---------------+----------------------+--------------------------------------------------+
   | **Parameter** | **Type**             | **Description**                                  |
   +===============+======================+==================================================+
   | ``in``        | ``mli_tensor *``     | [IN] Pointer to constant input tensor            |
   +---------------+----------------------+--------------------------------------------------+
   | ``cfg``       | ``mli_pool_cfg *``   | [IN] Pointer to pooling parameters structure     |
   +---------------+----------------------+--------------------------------------------------+
   | ``out``       | ``mli_tensor *``     | [IN | OUT] Pointer to output feature map tensor. |
   |               |                      | Result is stored here                            |
   +---------------+----------------------+--------------------------------------------------+
..


.. table:: List of Available Average Pooling Functions
   :align: center
   :widths: auto
   
   +-------------------------------------+-------------------------------+
   | **Function Name**                   | **Details**                   |
   +=====================================+===============================+
   | ``mli_krn_avepool_hwc_sa8``         || In/out layout: **HWC**       |
   |                                     || In/out data format: **sa8**  |
   |                                     || Supports any kernel size     |
   +-------------------------------------+-------------------------------+
   | ``mli_krn_avepool_hwc_fx16``        || In/out layout: **HWC**       |
   |                                     || In/out data format: **fx16** |
   |                                     || Supports any kernel size     |
   +-------------------------------------+-------------------------------+
   | ``mli_krn_avepool_hwc_sa8_k2x2``    || In/out layout: **HWC**       |
   |                                     || In/out data format: **sa8**  |
   |                                     || Kernel width: **2**          |
   |                                     || Kernel height: **2**         |
   +-------------------------------------+-------------------------------+
   | ``mli_krn_avepool_hwc_fx16_k2x2``   || In/out layout: **HWC**       |
   |                                     || In/out data format: **sa8**  |
   |                                     || Kernel width: **2**          |
   |                                     || Kernel height: **2**         |
   +-------------------------------------+-------------------------------+
   | ``mli_krn_avepool_hwc_sa8_k3x3``    || In/out layout: **HWC**       |
   |                                     || In/out data format: **sa8**  |
   |                                     || Kernel width: **3**          |
   |                                     || Kernel height: **3**         |
   +-------------------------------------+-------------------------------+
   | ``mli_krn_avepool_hwc_fx16_k3x3``   || In/out layout: **HWC**       |
   |                                     || In/out data format: **sa8**  |
   |                                     || Kernel width: **3**          |
   |                                     || Kernel height: **3**         |
   +-------------------------------------+-------------------------------+
..

Conditions
^^^^^^^^^^

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the used version of the kernel.

 - Shapes of ``in``  and  ``out`` tensors must be compatible,
   which implies the following requirements:

   - ``in`` and ``out`` are 3-dimensional tensors (rank==3). Dimensions meaning, 
     and order (layout) is aligned with the used version of kernel.

   - Channel :math:`Ci` dimension of ``in`` and ``out`` tensors must be equal.

   - Shapes of ``in`` and ``out`` tensors together with ``cfg`` structure 
     must satisfy the equations :eq:`eq_pool_shapes`

 - ``in`` and ``out`` tensors must not point to overlapped memory regions.

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

 - ``padding_top`` and ``padding_bottom`` parameters must be in range of [0, ``kernel_height``).

 - ``padding_left`` and ``padding_right`` parameters must be in range of [0, ``kernel_width``).

 - ``stride_width`` and ``stride_height`` parameters must not be equal to 0.

 - For sa8, ``in`` and ``out`` tensors must be quantized on the tensor level. This implies that 
   each tensor contains a single scale factor and a single zero offset.

 - For sa8, zero offset of in and out tensors must be within [-128, 127] range.

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the rest fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.
