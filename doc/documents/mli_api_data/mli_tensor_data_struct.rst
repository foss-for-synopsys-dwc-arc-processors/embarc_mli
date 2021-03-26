.. _mli_tens_data_struct:

mli_tensor Data Structures
--------------------------

All functions use the ``mli_tensor`` structure as the main container for input and output data. 
It represents a multi-dimensional array of elements. The ``mli_tensor`` struct describes the 
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

where 

``mli_element_type`` is defined as follows:   

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

.. _c_mli_data_container:

and ``mli_data_container`` is defined as follows:

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


Table :ref:`mli_tnsr_struc` describes the fields in the mli_tensor structure.

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
   | ``data``          | ``mli_data_container`` | - ``rank  > 0``: General Tensor. The tensor contains a pointer to the       |
   |                   |                        |   data.                                                                     |
   |                   |                        |                                                                             |
   |                   |                        | - ``rank == 0``: Scalar tensor. The tensor holds only a single value and    |
   |                   |                        |   this value is directly stored into this field.                            |
   |                   |                        |                                                                             |
   |                   |                        | This field has a union of different possible data container types. For      |
   |                   |                        | scalar tensors (tensors with a single element), this field is not a         |
   |                   |                        | pointer, but it contains the data itself.                                   |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``data.capacity`` | ``uint32_t``           | Size in bytes of the allocated memory that the data field points to. In     |
   |                   |                        | case there is no buffer attached (``rank == 0``), the capacity is set to 0. |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``shape``         | ``uint32_t[]``         | Array with tensor dimensions. Dimensions are stored in order starting from  |
   |                   |                        | the one with the largest stride between the data portions.                  |
   |                   |                        | For example, for tensor T of size (channels, height width) stored in HWC    |
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
   |                   |                        | is defined by ``MLI_MAX_RANK*``.If the mem_stride is set to 0, it is        |
   |                   |                        | computed from the shape.                                                    |
   |                   |                        |                                                                             |
   |                   |                        | Manually-set values of mem_stride array must decrease gradually and must    |
   |                   |                        | not be less than if they would be computed from the shape. For example,     |
   |                   |                        | for a tensor of shape [Height, Width, Channels):                            |
   |                   |                        |                                                                             |
   |                   |                        |  - mem_stride[0] >= 1 x Channels x Width AND mem_stride[0] >= mem_stride[1] |
   |                   |                        |                                                                             |
   |                   |                        |  - mem_stride[1] >= 1*Channels    AND mem_stride[1] >= mem_stride[2]        |
   |                   |                        |                                                                             |
   |                   |                        |  - mem_stride[2] >= 1.                                                      |
   |                   |                        |                                                                             |
   |                   |                        | In case the mem_stride is computed from the shape, then the kernel will not |
   |                   |                        | update this field in the tensor struct. The only exception is the           |
   |                   |                        | mli_move function, which can write the mem_stride field of the dst tensor.  |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``rank``          | ``uint32_t``           | Number of dimensions of this tensor (Must be less or equal to               |
   |                   |                        | ``MLI_MAX_RANK*``)                                                          |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``el_type``       | ``mli_element_type``   | Enum depicting the type of the element stored in the tensor. Supported      |
   |                   |                        | values in this enum are listed previously. For details, see :ref:`data_fmts`|
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``el_params``     | ``mli_element_params`` | Union of structs containing the quantization parameters of the elements     |
   |                   |                        | stored in the tensor.  Details on supported quantization schemes are        |
   |                   |                        | discussed in :ref:`data_fmts`                                               |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
     
..

\* ``MLI_MAX_RANK`` is set to 4.

:ref:`t_mli_el_p_union` describes the fields in the mli_element_params union.  Several members of this union 
are used to support per-axis quantization. ``sa.dim`` indicates over which axis (dimension) of the tensor the 
quantization parameters can vary. For instance in a CHW layout, dim = 0 means that for each channel there is 
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
   |                        |                        | Only MLI_EL_PARAM_SC16_ZP16 is currently supported which reflects the       |
   |                        |                        | following parameters according the description below.                       |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.zeropoint``       | ``mli_data_container`` | 16-bit signed zero-point offset.                                            |
   |                        |                        |                                                                             |
   |                        |                        | - ``sa.dim < 0``: Single value for all data in tensor.                      |
   |                        |                        |                                                                             |
   |                        |                        | - ``sa.dim >= 0``: Pointer to an array of zero points relating to           |
   |                        |                        |   configured dimension (``sa.dim``).                                        |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.scale``           | ``mli_data_container`` | 16-bit signed scale factors. Only positive scale factors are supported.     |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim < 0``: ``sa.scale`` is a single value for all data in tensor  |
   |                        |                        |                                                                             |
   |                        |                        | - If ``sa.dim >= 0``:  ``sa.scale`` is a pointer to an array of             |
   |                        |                        |   scale factors related to configured dimension (``sa.dim``).               |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.dim``             | ``int32_t``            | Tensor dimension to which the arrays of quantization parameters apply       |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
   | ``sa.scale_frac_bits`` | ``int32_t``            | ``sa.scale`` is an array of fixed point scale values. This field contains   |
   |                        |                        | the (shared) exponent of these values, stored as the number of fractional   |
   |                        |                        | bits for the elements in the scales array.                                  |
   +------------------------+------------------------+-----------------------------------------------------------------------------+
..
   
   
   