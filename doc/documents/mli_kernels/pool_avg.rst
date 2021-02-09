Average Pooling Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Average pooling computes the average value of all the values in the kernel for each 
output element. The kernel is defined as the area of the kernel width x kernel height. 

Kernels which implement average pooling functions have the following prototype:

.. code::

   mli_status  mli_krn_avepool_hwc_<data_format>(
      const mli_tensor *in,
      const mli_pool_cfg *cfg,
      mli_tensor *out);
..

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto
   	  
   +---------------+----------------------+-----------------------------------------------+
   | **Parameter** | **Type**             | **Description**                               |
   +===============+======================+===============================================+
   | ``in``        | ``mli_tensor *``     | [IN] Pointer to constant input tensor.        |
   +---------------+----------------------+-----------------------------------------------+
   | ``cfg``       | ``mli_pool_cfg *``   | [IN] Pointer to pooling parameters structure  |
   +---------------+----------------------+-----------------------------------------------+
   | ``out``       | ``mli_tensor *``     | [OUT] Pointer to output feature map tensor.   |
   |               |                      | Result is stored here                         |
   +---------------+----------------------+-----------------------------------------------+
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

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity and 
   valid ``el_params`` union. Other fields of the structure do not have to contain valid 
   data and are filled by the function.
	
 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
 - ``padding_top`` and ``padding_bottom`` parameters must be in range of (0, weights (H)eight).
 
 - ``padding_left`` and ``padding_right`` parameters must be in range of (0, weights (W)idth).
 
 - ``stride_width`` and ``stride_height`` parameters must not be equal to 0.
 
For sa8, ``in`` and ``out`` tensor must be quantized on the tensor level. It implies that each 
tensor contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.
   