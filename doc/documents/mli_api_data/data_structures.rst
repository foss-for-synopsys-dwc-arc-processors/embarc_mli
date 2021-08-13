Data Structures
---------------

This section describes the set of available data structures present in the MLI interface:

 - :ref:`mli_tens_data_struct` - The main container to be passed for input and output data used 
   by functions in MLI Interface
 - :ref:`c_mli_data_container` - The container to handle raw data arrays or values.
 - :ref:`c_mli_element_type` - The type of the element stored in the ``mli_tensor``. 
 - :ref:`c_mli_element_params` - Quantization parameters of the elements stored in the ``mli_tensor``. 
 - :ref:`mli_lut_data_struct` - Look-Up table handler used in transformation kernels. 
 - :ref:`kernl_sp_conf` 


.. _mli_tens_data_struct:

``mli_tensor`` Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All functions use the ``mli_tensor`` structure as the main container for input and output data. 
It represents a multi-dimensional array of elements. The ``mli_tensor`` structure describes the 
shape of this array, its data format, and the way it is organized in memory.

.. code:: c

   typedef struct mli_tensor {
      mli_data_container data;
      uint32_t shape[MLI_MAX_RANK];
      int32_t mem_stride[MLI_MAX_RANK];
      uint32_t rank;
      mli_element_type el_type;
      mli_element_params el_params;
   } mli_tensor;
..

The table :ref:`mli_tnsr_struc` describes the fields in the ``mli_tensor`` structure.

.. _mli_tnsr_struc:  
.. table:: mli_tensor Structure Field Descriptions
   :align: center
   :widths: 50, 50, 130 
   
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | **Field name**    | **type**               | **Comment**                                                                 |
   +===================+========================+=============================================================================+
   |                   |                        | The meaning of this field varies based on the setting of the ``rank``       |
   |                   |                        | field:                                                                      |
   |                   |                        |                                                                             |
   | ``data``          | ``mli_data_container`` | - ``rank  > 0``: General Tensor. The ``data`` contains a pointer to the     |
   |                   |                        |   data.                                                                     |
   |                   |                        |                                                                             |
   |                   |                        | - ``rank == 0``: Scalar tensor. The ``data`` holds only a single value and  |
   |                   |                        |   this value is directly stored into this field.                            |
   |                   |                        |                                                                             |
   |                   |                        | Type of the data preserved in this container must be in sync                |
   |                   |                        | with ``el_type`` field.                                                     |
   |                   |                        |                                                                             |
   |                   |                        | See :ref:`c_mli_data_container` for more info.                              |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``shape``         | ``uint32_t[]``         | Array with tensor dimensions. Dimensions are stored in order starting from  |
   |                   |                        | the one with the largest stride between the data portions.                  |
   |                   |                        | For example, for tensor T of shape (channels, height width) stored in HWC   |
   |                   |                        | layout, shape[0] = height, shape[1] = width, shape[2] = channels. Shape[3]  |
   |                   |                        | is unused. The size of the array is defined by ``MLI_MAX_RANK*``.           |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``mem_stride``    | ``int32_t[]``          | Array with the distance (in elements) to the next element in the same       |
   |                   |                        | dimension. Positive values are supported only.                              |
   |                   |                        | To compute the size in bytes, the number of elements needs to be            |
   |                   |                        | multiplied by the bytes per element. For example, for a matrix              |
   |                   |                        | A(rows,columns), ``mem_stride[1]`` contains the distance to the next        |
   |                   |                        | element (=1 in this example), and ``mem_stride[0]`` contains the distance   |
   |                   |                        | from one row to the next (=columns in this example). The size of the array  |
   |                   |                        | is defined by ``MLI_MAX_RANK*``.                                            |
   |                   |                        |                                                                             |
   |                   |                        | Values of ``mem_stride`` array must decrease gradually and                  |
   |                   |                        | must not be less than if they would be computed from the shape. For         |
   |                   |                        | example, for a tensor of shape :math:`(Height, Width, Channels)`:           |
   |                   |                        |                                                                             |
   |                   |                        |  - ``mem_stride[0] >= 1 x Channels x Width``                                |
   |                   |                        |    AND ``mem_stride[0] >= mem_stride[1]``                                   |
   |                   |                        |                                                                             |
   |                   |                        |  - ``mem_stride[1] >= 1*Channels`` AND ``mem_stride[1] >= mem_stride[2]``   |
   |                   |                        |                                                                             |
   |                   |                        |  - ``mem_stride[2] >= 1``                                                   |
   |                   |                        |                                                                             |
   |                   |                        | ``mli_move`` is the only function which can write the ``mem_stride`` field  |
   |                   |                        | of the ``dst`` tensor. Other kernels don't update this field                |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``rank``          | ``uint32_t``           | Number of dimensions of this tensor (Must be less or equal to               |
   |                   |                        | ``MLI_MAX_RANK*``)                                                          |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``el_type``       | ``mli_element_type``   | Enum depicting the type of the element stored in the tensor.                |
   |                   |                        | See :ref:`c_mli_element_type` for more info.                                |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``el_params``     | ``mli_element_params`` | Union of structs containing the quantization parameters of the elements     |
   |                   |                        | stored in the tensor.  Details on supported quantization schemes are        |
   |                   |                        | discussed in :ref:`data_fmts`                                               |
   |                   |                        |                                                                             |
   |                   |                        | See :ref:`c_mli_element_params` for more info.                              |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
     
..

.. note::
   ``MLI_MAX_RANK`` is set to 4.
..

.. important::
   ``mli_tensor`` is valid if all its fields are populated in a non-contradictory way which implies: 

      - ``rank`` and ``shape`` fields are aligned with ``mem_strides`` field
      - ``data`` container points to a memory region of ``el_type`` elements or contains a single 
        element itself (see :ref:`c_mli_data_container`). Its capacity is enough to preserve data described 
        by ``rank``, ``shape`` and ``mem_stride`` fields.
      - ``el_params`` structure is filled properly according to ``el_type`` field and
        related quantization scheme (see :ref:`data_fmts`)
..

.. _c_mli_data_container:

``mli_data_container`` Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mli_data_container`` is a container to represent polymorphic data. 
It stores pointer to data or a single value that intend to be directly used in arithmetical operations.

``mli_data_container`` is defined as follows:

.. code:: c
 
   typedef struct _mli_data_container {
     uint32_t  capacity;
     union {
       int32_t*  pi32;
       int16_t*  pi16;
       int8_t*   pi8;
       float*    pf32;
       int32_t   i32;
       int16_t   i16;
       int8_t    i8;
       float     f32;
     } mem;
   } mli_data_container;
..

:ref:`t_mli_d_cont_strct` describes the fields in the ``mli_data_container`` struture. 

.. _t_mli_d_cont_strct:
.. table:: mli_data_container Structure Field Description
   :align: center
   :widths: 50, 50, 130 
   
   +--------------------+------------------+----------------------------------------------------------------------+
   | **Field Name**     | **Type**         | **Comment**                                                          |
   +====================+==================+======================================================================+
   | ``capacity``       | ``uint32_t``     | Size in bytes of the memory that the ``mem`` field points to.        |
   |                    |                  | In case there is no buffer attached, the capacity must be set to 0.  |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem``            | Union            | This field is the union of different possible data container types.  |
   |                    |                  | In case capacity is set to 0, this field is not a pointer,           |
   |                    |                  | but it contains the data itself.                                     |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.pi32``       | ``int32_t *``    | Pointer to array of 32 bit signed integer values.                    |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.pi16``       | ``int16_t *``    | Pointer to array of 16 bit signed integer values                     |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.pi8``        | ``int8_t *``     | Pointer to array of 8 bit signed integer values                      |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.pf32``       | ``float *``      | Pointer to array of 32bit single precision floating point value      |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.i32``        | ``int32_t``      | 32 bit signed integer value.                                         |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.i16``        | ``int16_t``      | 16 bit signed integer value.                                         |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.i8``         | ``int8_t``       | 8 bit signed integer value.                                          |
   +--------------------+------------------+----------------------------------------------------------------------+
   | ``mem.f32``        | ``float``        | 32bit single precision floating point value.                         |
   +--------------------+------------------+----------------------------------------------------------------------+
   
..


.. _c_mli_element_type:

``mli_element_type`` Enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mli_element_type`` enumeration defines the basic element type stored in tensor structure.
Based on this information, library functions may define sizes, algorithms for processing,
and other  implementation specific things. ``mli_element_type`` is defined as follows:   

.. code:: c

  typedef enum {
     MLI_EL_FX_4  = 0x004,
     MLI_EL_FX_8  = 0x008,
     MLI_EL_FX_16 = 0x010,
     MLI_EL_SA_8  = 0x108,
     MLI_EL_SA_32 = 0x120,
     MLI_EL_FP_16 = 0x210,
     MLI_EL_FP_32 = 0x220
  } mli_element_type;
..

:ref:`t_mli_el_type` describes the entities in the ``mli_element_type`` union. 

.. _t_mli_el_type:
.. table:: mli_element_type Enumeration Values Description
   :align: center
   :widths: 50, 50, 130 

   +-----------------------+----------------------------------------------------------------------------+
   | **Enumeration Value** | **Description**                                                            |
   +=======================+============================================================================+
   | ``MLI_EL_FX_4``       | 4 bit depth fixed point data. For future use.                              |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_FX_8``       | 8 bit depth fixed point data. See ``fx8`` in :ref:`mli_data_fmts`          |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_FX_16``      | 16 bit depth fixed point data. See ``fx16`` in :ref:`mli_data_fmts`        |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_SA_8``       | 8 bit asymetrical signed data. See ``sa8`` in :ref:`mli_data_fmts`         |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_SA_32``      | 32 bit asymetrical signed data. See ``sa32`` in :ref:`mli_data_fmts`       |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_FP_16``      | Half precision floating point data. For future use.                        |
   +-----------------------+----------------------------------------------------------------------------+
   | ``MLI_EL_FP_32``      | Single precision floating point data. See ``fp32`` in :ref:`mli_data_fmts` |
   +-----------------------+----------------------------------------------------------------------------+

..


.. _c_mli_element_params:

``mli_element_params`` Union
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mli_element_params`` stores data quantization parameters required for arithmetical 
operations with tensor elements. Details on supported quantization schemes are discussed in :ref:`data_fmts`.

``mli_element_params`` is defined as follows:

.. code:: c
 
   typedef union _mli_element_params {
      struct{
         uint32_t frac_bits;
      } fx; /* FiXed point \*/
  
      struct {
         mli_el_param_type type;
         mli_data_container zero_point;
         mli_data_container scale;
         mli_data_container scale_frac_bits;
         int32_t dim;
      } sa; /* Signed Asymmetric \*/
   } mli_element_params;
..

``mli_el_param_type`` is defined as follows:

.. code:: c
 
   typedef enum {
      MLI_EL_PARAM_SC16_ZP16 = 0
   } mli_el_param_type;
..

:ref:`t_mli_el_p_union` describes the fields in the ``mli_element_params`` union.  Several members of this union 
are used to support per-axis quantization. ``sa.dim`` indicates over which axis (dimension) of the tensor the 
quantization parameters can vary. For instance in a CHW layout, ``dim`` = 0 means that for each channel there is 
a different zero point and a different scale factor. The size of these arrays is the same as the number of 
channels in the tensor ``(array_size = shape[dim])``.

.. _t_mli_el_p_union:
.. table:: mli_element_params Union Field Description
   :align: center
   :widths: 50, 50, 130 
   
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | **Field Name**         | **Type**               | **Comment**                                                                 |
   +========================+========================+=============================================================================+
   | ``fx.frac_bits``       | ``uint8_t``            | Number of fractional bits.                                                  |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.type``            | ``mli_el_param_type``  | Enum depicting the types of the quantization parameters in the tensor.      |
   |                        |                        | Only ``MLI_EL_PARAM_SC16_ZP16`` is currently supported which reflects the   |
   |                        |                        | following parameters according the description below.                       |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.dim``             | ``int32_t``            | Tensor dimension to which the arrays of quantization parameters apply       |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.zeropoint``       | ``mli_data_container`` | 16-bit signed integer zero-point offset.                                    |
   |                        |                        |                                                                             |
   |                        |                        | - ``sa.dim < 0``: Single value for all data in tensor.                      |
   |                        |                        |                                                                             |
   |                        |                        | - ``sa.dim >= 0``: Pointer to an array of zero points relating to           |
   |                        |                        |   configured dimension (``sa.dim``).                                        |
   |                        |                        |                                                                             |
   |                        |                        | See :ref:`c_mli_data_container` for more info.                              |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.scale``           | ``mli_data_container`` | 16-bit signed integer scale factors. Only positive scale factors are        |
   |                        |                        | supported.                                                                  |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim < 0``: ``sa.scale`` is a single value for all data in tensor  |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim >= 0``:  ``sa.scale`` is a pointer to an array of             |
   |                        |                        |   scale factors related to configured dimension (``sa.dim``).               |
   |                        |                        |                                                                             |
   |                        |                        | See :ref:`c_mli_data_container` for more info.                              |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.scale_frac_bits`` | ``mli_data_container`` | 8-bit signed integer exponent of values in ``sa.scale`` field. The field    |
   |                        |                        | stores the exponent as the number of fractional bits.                       |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim < 0``: ``sa.scale_frac_bits`` is a single value               |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim >= 0``:  ``sa.scale_frac_bits`` is a pointer to an array of   |
   |                        |                        |   frac bits per each value in ``sa.scale`` array.                           |
   |                        |                        |                                                                             |
   |                        |                        | See :ref:`c_mli_data_container` for more info.                              |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   
..

.. admonition:: Example 
   :class: "admonition tip"

   FX16 tensor might be populated in the following way:

   .. code:: c

       mli_tensor tsr_fx16 = {0};

       // Filling quantization params
       tsr_fx16.el_type = MLI_EL_FX_16;
       tsr_fx16.el_params.fx.frac_bits = 12;

       // Filling other fields of tsr_fx16
       ...
   ..

   SA8 tensor quantized on per-tensor level might be populated in the following way:

   .. code:: c

      mli_tensor tsr_sa8 = {0};
      
      // Filling quantization params
      tsr_sa8.el_type = MLI_EL_SA8;
      tsr_sa8.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
      tsr_sa8.el_params.sa.dim = -1; // e.g Per-Tensor (all values shares the same quant params)
   
      // Set all capacities to 0 as values are directly stored inside the container
      tsr_sa8.el_params.sa.zero_point.capacity = 0;
      tsr_sa8.el_params.sa.scale_frac_bits.capacity = 0;
      tsr_sa8.el_params.sa.scale.capacity = 0;

      tsr_sa8.el_params.sa.zero_point.mem.i16 = -128;
      tsr_sa8.el_params.sa.scale_frac_bits.mem.i8 = 3;
      tsr_sa8.el_params.sa.scale.mem.i16 = 5; // (5 \ 2^3) = 0.625

      // Filling other fields of tsr_sa8
      ...

   ..

   SA8 tensor quantized on per-axis level might be populated in the following way:

   .. code:: c

      mli_tensor tsr_sa8_per_axis = {0};
      int16_t scales[] = {...};
      int8_t scales_frac[] = {...};
      int16_t zero_points[] = {...};
      
      // Filling quantization params
      tsr_sa8_per_axis.el_type = MLI_EL_SA8;
      tsr_sa8_per_axis.el_params.sa.type = MLI_EL_PARAM_SC16_ZP16;
      tsr_sa8_per_axis.el_params.sa.dim = 0; // e.g Per 0th dimension
      
      tsr_sa8_per_axis.el_params.sa.zero_point.mem.pi16 = zero_points;
      tsr_sa8_per_axis.el_params.sa.zero_point.capacity = sizeof(zero_points);

      tsr_sa8_per_axis.el_params.sa.scale_frac_bits.mem.pi8 = scales_frac;
      tsr_sa8_per_axis.el_params.sa.scale_frac_bits.capacity = sizeof(scales_frac);

      tsr_sa8_per_axis.el_params.sa.scale.mem.pi16 = scales;
      tsr_sa8_per_axis.el_params.sa.scale.capacity = sizeof(scales);

      // Filling other fields of tsr_sa8_per_axis
      ...
   ..
..


.. _mli_lut_data_struct:

``mli_lut`` Structure
~~~~~~~~~~~~~~~~~~~~~~

Several functions use a look-up table (LUT) to perform data transformation.  The LUT represents a function in a 
table form that can be used to transform input values (function argument) to output values (function result). 
The ``mli_lut`` structure is a representation of such a table.

The ``mli_lut`` struct describes the data in the LUT, including the format of its input and output.

.. code:: c

   typedef struct _mli_lut{
      mli_data_container data;
      mli_element_type type;
      int32_t length;
      int32_t in_frac_bits;
      int32_t out_frac_bits;
      int32_t input_offset;
      int32_t output_offset;
   } mli_lut;
..

The following table describes the fields in the ``mli_lut`` structure.
   
.. _mli_lut_struct_table:  
.. table:: mli_lut Structure Field Descriptions
   :align: center
   :widths: 50, 50, 130 
   
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | **Field name**    | **type**               | **Comment**                                                                 |
   +===================+========================+=============================================================================+
   |                   |                        | This field has a union of different possible data container types.          |
   |   ``data``        | ``mli_data_container`` | Pointer of specified type (see the type field in this table) should point   |
   |                   |                        | to an array with the LUT table data.                                        |
   |                   |                        |                                                                             |
   |                   |                        | See :ref:`c_mli_data_container` for more info.                              |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``data.capacity`` | ``uint32_t``           | Size in bytes of the allocated memory that the data field points to.        |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``type``          | ``mli_element_type``   | Enum depicting the type of the element stored in the data field.            |
   |                   |                        | Values in this enum are listed in section :ref:`c_mli_element_type`.        |
   |                   |                        | Only ``MLI_EL_FX_16`` entity is currently supported.                        |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``length``        | ``int32_t``            | Number of values stored in the LUT table                                    |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``in_frac_bits``  | ``int32_t``            | Number of fractional bits for the LUT input (argument)                      |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``out_frac_bits`` | ``int32_t``            | Number of fractional bits for the LUT output (result)                       |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``input_offset``  | ``int32_t``            | Offset of input argument which is added before applying the LUT function.   |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``output_offset`` | ``int32_t``            | Offset of output which is subtracted from LUT function result.              |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
     
..


.. _kernl_sp_conf:

Kernel Specific Configuration Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A significant number of MLI kernels must be configured by specific parameters, which 
influence calculations and results, but are not directly related to input data. For 
example, padding and stride values are parameters of the convolution layer and the type 
of ReLU is a parameter for ReLU transform layer. All specific parameters for 
particular primitive type are grouped into structures. This document describes these 
structures along with the kernel description they relate to. The following tables 
describe fields of existing MLI configuration structures:

 - Table :ref:`t_mli_conv2d_cfg_desc`
 
 - Table :ref:`t_mli_fc_cfg_desc` 

 - Table :ref:`t_mli_rnn_cell_cfg_desc` 

 - Table :ref:`t_mli_rnn_dense_cfg_desc`

 - Table :ref:`t_mli_pool_cfg_desc` 

 - Table :ref:`t_mli_argmax_cfg_desc`

 - Table :ref:`t_mli_permute_cfg_desc`

 - Table :ref:`t_mli_relu_cfg_desc`

 - Table :ref:`t_mli_prelu_cfg_desc`

 - Table :ref:`t_mli_mov_cfg_desc`

..
   - Table :ref:`t_mli_sub_tensor_cfg_desc`

