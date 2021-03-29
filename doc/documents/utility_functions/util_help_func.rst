Helper Functions Group
----------------------

The following Helper functions are used for 
getting information from data structures and performing various operations on them.

 - :ref:`get_elem_size`
 
 - :ref:`count_elements`
 
 - :ref:`get_scale_val`

 - :ref:`get_shift_val`
 
 - :ref:`get_zero_offset_val`

..
   - :ref:`point_sub_tensor`

 - :ref:`num_of_accu_bits`
 
 
.. _get_elem_size:

Get Element Size
~~~~~~~~~~~~~~~~

This function returns the size of the tensor basic element in bytes. It returns 0 if the in pointer 
does NOT point to a tensor with a supported element type (see description of mli_element_type 
in section :ref:`kernl_sp_conf`).

Function prototype:

.. code:: c

   uint32_t mli_hlp_tensor_element_size(
       const mli_tensor *in);
..

The parameters are described in :ref:`t_mli_hlp_count_elem_num_params`.

.. _t_mli_hlp_get_elem_size:
.. table:: mli_hlp_tensor_element_size Parameters
   :align: center
   :widths: auto
   
   +--------------------+-----------------+-------------------------------------+
   | **Field Name**     | Type            | Description                         |
   +====================+=================+=====================================+
   | ``in``             | ``mli_tensor*`` | [IN] Pointer to input tensor        |
   +--------------------+-----------------+-------------------------------------+
..

Conditions:
 - ``in`` must point to a tensor structure

.. _count_elements:

Count Number of Elements
~~~~~~~~~~~~~~~~~~~~~~~~

This function counts the number of elements in a tensor starting from the provided dimension 
number (dimension numbering starts from 0): 

.. math::

   elementCount=shape_{startdim} * shape_{(startdim+1)}*… *shape_{(rank-1)}
..

When used with :math:`startdim = 0`, the total element count of the tensor is computed.

Function prototype:

.. code:: c

   uint32_t mli_hlp_count_elem_num(
       const mli_tensor *in,
       uint32_t start_dim);
..

The parameters are described in :ref:`t_mli_hlp_count_elem_num_params`.

.. _t_mli_hlp_count_elem_num_params:
.. table:: mli_hlp_count_elem_num Parameters
   :align: center
   :widths: auto
   
   +--------------------+-----------------+-------------------------------------+
   | **Field Name**     | Type            | Description                         |
   +====================+=================+=====================================+
   | ``in``             | ``mli_tensor*`` | [IN] Pointer to input tensor        |
   +--------------------+-----------------+-------------------------------------+
   | ``start_dim``      | ``uint32_t``    | [IN] Start dimension for counting   |
   +--------------------+-----------------+-------------------------------------+
..

Conditions:

 - ``in`` must contain a valid rank (less than or equal to ``MLI_MAX_RANK``)

 - ``start_dim`` must be less than the input rank

.. _get_scale_val:
 
Get Scale Value
~~~~~~~~~~~~~~~

This function returns the scale value from the quantization parameters. For data 
formats that don’t have a scale value, the value 1 is returned. 
For tensors with multiple scale value per-axis scale_idx parameter defines the 
particular scale value to be fetched. In case of an invalid tensor, the value 0 is returned.

Function prototype:

.. code:: c

   int32_t mli_hlp_tensor_scale(
      const mli_tensor *in
      const uint32_t scale_idx
   );
..
  
The parameters are described in Table :ref:`t_mli_hlp_tensor_scale_params`.
 
.. _t_mli_hlp_tensor_scale_params:
.. table:: mli_hlp_tensor_scale Parameters
   :align: center
   :widths: auto
   
   +----------------+-----------------+-------------------------------------------------------+
   | **Field name** | **Type**        | **Description**                                       |
   +================+=================+=======================================================+
   | ``in``         | ``mli_tensor*`` | [IN] Pointer to input tensor                          |  
   +----------------+-----------------+-------------------------------------------------------+ 
   | ``scale_idx``  | ``uint32_t``    | [IN] Index of a specific scale value from the tensor  |  
   +----------------+-----------------+-------------------------------------------------------+ 
..   

Conditions:

 - ``in`` must contain a valid data format
 - ``scale_idx`` must be less or equal to number of scale values in the tensor

.. _get_shift_val:
 
Get Scale Shift Value
~~~~~~~~~~~~~~~~~~~~~

This function returns the shift value from the quantization parameters. 
For data formats that don't have a shift value, the value 0 is returned.
For tensors with multiple scale values per-axis, the parameter ``scale_idx`` 
defines the particular scale shift value to be fetched.

Function prototype

.. code:: c

   int32_t mli_hlp_tensor_scale_shift(
       const mli_tensor *in
       const uint32_t scale_idx
   );
..
	  
The parameters are described in Table :ref:`t_mli_hlp_tensor_scale_shift_params`

.. _t_mli_hlp_tensor_scale_shift_params:
.. table:: mli_hlp_tensor_scale_shift Parameters
   :align: center
   :widths: auto
   
   +----------------+-----------------+-------------------------------------------------------------+
   | **Field name** | **Type**        | **Description**                                             |
   +================+=================+=============================================================+
   | ``in``         | ``mli_tensor*`` | [IN] Pointer to input tensor                                |  
   +----------------+-----------------+-------------------------------------------------------------+
   | ``scale_idx``  | ``uint32_t``    | [IN] Index of a specific scale shift value from the tensor  |  
   +----------------+-----------------+-------------------------------------------------------------+ 
.. 

Conditions:

 - ``in`` must contain a valid data format
 - ``scale_idx`` must be less or equal to number of scale values in the tensor

.. _get_zero_offset_val:
 
Get Zero Offset Value
~~~~~~~~~~~~~~~~~~~~~

This function returns the zero offset value from the quantization parameters.
For data formats that do not have a zero offset value, the value 0 is returned.
For tensors with multiple zero offset values per-axis, the parameter ``scale_idx`` 
defines the particular zero offset value to be fetched.

Function prototype:

.. code:: c

   int16_t mli_hlp_tensor_zero_offset(
       const mli_tensor *in
       const uint32_t zero_idx
   );
..
  
The parameters are described in Table :ref:`t_mli_hlp_tensor_zero_offset_params`.

.. _t_mli_hlp_tensor_zero_offset_params:
.. table:: mli_hlp_tensor_zero_offset Parameters
   :align: center
   :widths: auto
   
   +----------------+-----------------+-------------------------------------------------------------+
   | **Field name** | **Type**        | **Description**                                             |
   +================+=================+=============================================================+
   | ``in``         | ``mli_tensor*`` | [IN] Pointer to input tensor                                |  
   +----------------+-----------------+-------------------------------------------------------------+ 
   | ``zero_idx``   | ``uint32_t``    | [IN] Index of a specific zero offset value from the tensor  |  
   +----------------+-----------------+-------------------------------------------------------------+ 
.. 

Conditions:

 - ``in`` must contain a valid data format
 - zero_idx must be less or equal to number of zero offset values in the tensor
 
.. _point_sub_tensor:

..
   Point to Sub-Tensor
   ~~~~~~~~~~~~~~~~~~~

   .. warning::

      The interface of this function is subject to change. Avoid using it.

   ..

   This function points to sub tensors in the input tensor. This function can 
   be considered as indexing in a multidimensional array without copying or 
   used to create a slice/fragment of the input tensor without copying the data.

   For example, given a HWC tensor, this function could be used to create a HWC 
   tensor for the top half of the HW image for all channels.

   The configuration struct is defined as follows and the fields are explained in 
   Table :ref:`t_mli_sub_tensor_cfg_desc`.

   .. code:: c

      typedef struct {
      uint32_t offset[MLI_MAX_RANK];
      uint32_t size[MLI_MAX_RANK];
      uint32_t sub_tensor_rank;
      } mli_sub_tensor_cfg;
   ..

   .. _t_mli_sub_tensor_cfg_desc:
   .. table:: mli_sub_tensor_cfg Structure Field Description
      :align: center
      :widths: auto
      
      +---------------------+----------------+---------------------------------------------------------+
      | **Field Name**      | **Type**       | Description                                             |
      +=====================+================+=========================================================+
      |                     |                | Start coordinate in the input tensor. Values must       |
      | ``offset``          | ``uint32_t[]`` | be smaller than the shape of the input tensor. Size     |
      |                     |                | of the array must be equal to the rank of the input     |
      |                     |                | tensor.                                                 |
      +---------------------+----------------+---------------------------------------------------------+
      |                     |                | Size of the sub tensor in elements per dimension:       |
      | ``size``            | ``uint32_t[]`` |                                                         |
      |                     |                | Restrictions:  Size[d] +   offset[d] <= input->shape[d] |
      +---------------------+----------------+---------------------------------------------------------+
      |                     |                | Rank of the sub tensor that is produced. Must be        |
      |                     |                | smaller or equal to the rank of the input tensor. If    |
      | ``sub_tensor_rank`` | ``uint32_t``   | the ``sub_tensor_rank`` is smaller than the input rank, |
      |                     |                | the dimensions with a size of 1 is removed in the       |
      |                     |                | output shape starting from the first dimension until    |
      |                     |                | the requested ``sub_tensor_rank`` value is reached.     |
      +---------------------+----------------+---------------------------------------------------------+ 
   ..

   This function computes the new data pointer based on the offset vector and it sets 
   the shape of the output tensor according to the size vector. The ``mem_stride`` fields 
   are copied from the input to the output, so after this operation, the output tensor might  
   not be a contiguous block of data.

   The function also reduces the rank of the output tensor if requested by the 
   configuration. Only the dimensions with a size of 1 can be removed. Data format and 
   quantization parameters are copied from the input to the output tensor.

   The capacity field of the output is the input capacity decremented with the same 
   value as that used to increment the data pointer.

   The function prototype:

   .. code:: c

      mli_status mli_hlp_subtensor(
      const mli_tensor *in,
      const mli_subtensor_cfg *cfg,
      mli_tensor *out);
   ..
   
   Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
   check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.


.. _num_of_accu_bits:
 
Get Number of Accumulator Guard Bits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions return the number of accumulator guard bits for a specific MAC (multiply-and-accumulate)
variant. An addition might result in an overflow if all bits of operands are used and both operands
hold the maximum (or minimum) values. It means that an extra bit is required for this operation.
But, if a sum of several operands is needed (accumulation), more than one extra bit is required to 
ensure that the result does not overflow. This function returns the number of such extra bits needed 
in the accumulation for MAC-based kernels. See :ref:`quant_accum_infl` section for more information.
Separate functions exist for each combination of input operands.

The function prototype:

.. code:: c

   uint8_t mli_hlp_accu_guard_bits_<operands>();
..

Where ``operands`` is a combination of input operands involved into MAC operation.

Here is a list of all available guard bits functions:

.. table:: List of Available Accum Guard Bits Functions
   :align: center
   :widths: auto 
   
   +---------------------------------------------+-----------------------------------------+
   | Function Name                               | Details                                 |
   +=============================================+=========================================+
   | ``mli_hlp_accu_guard_bits_sa8_sa8``         || Data format of both operands: **sa8**  |
   +---------------------------------------------+-----------------------------------------+
   | ``mli_hlp_accu_guard_bits_fx16_fx16``       || Data format of both operands: **fx16** |
   +---------------------------------------------+-----------------------------------------+
   | ``mli_hlp_accu_guard_bits_fx16_fx8``        || Data format of operands: **fx16** for  |
   |                                             || one and **fx8** another                |
   +---------------------------------------------+-----------------------------------------+

There are no specific requirements for ``mli_hlp_accu_guard_bits<operands>`` functions. 
These can be called at any time.
