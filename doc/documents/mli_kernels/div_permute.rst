.. _permute_prot:

Permute Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

The kernel permutes dimensions of input tensor according to provided order. In other words,
it transposes input tensors.

Functions
^^^^^^^^^

The functions which implement Permute have the following prototype:

.. code:: c

   mli_status mli_krn_permute_<data_format>(
      const mli_tensor *in,
      const mli_permute_cfg *cfg,	
      mli_tensor *out);	
..
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function parameters 
are shown in the following table:

.. table:: Permute Function Parameters
   :align: center
   :widths: auto
   
   +----------------+-------------------------+----------------------------------------------------------+
   | **Parameter**  | **Type**                | **Description**                                          |
   +================+=========================+==========================================================+
   | ``in``         | ``mli_tensor *``        | [IN] Pointer to constant input tensor                    |
   +----------------+-------------------------+----------------------------------------------------------+
   | ``cfg``        | ``mli_permute_cfg *``   | [IN] Pointer to Permute parameters structure             |
   +----------------+-------------------------+----------------------------------------------------------+
   | ``out``        | ``mli_tensor *``        | [OUT] Pointer to output tensor. Result is stored here    |
   +----------------+-------------------------+----------------------------------------------------------+
..

``mli_permute_cfg`` structure is defined as:

.. code:: c

   typedef struct {
      uint8_t perm_dim[MLI_MAX_RANK];
   }  mli_permute_cfg;
..

.. _t_mli_permute_cfg_desc:
.. table:: mli_permute_cfg Structure Field Description
   :align: center
   :widths: auto
   
   +-----------------+------------------+-------------------------------------------------------------+
   | **Field name**  | **Type**         | **Description**                                             |
   +=================+==================+=============================================================+
   | ``perm_dim``    | ``uint8_t[]``    | A permutation array. Dimensions order for output tensor.    |
   +-----------------+------------------+-------------------------------------------------------------+
..

The new order of dimensions is given by ``perm_dim`` array of kernel configuration structure. The 
``out`` tensor’s dimension ``idx`` corresponds to the dimension of the ``in`` tensor with ``perm_dim[idx]``. 
The tensor’s data is reordered according to the new shape.

For example, if input tensors have the shape (2, 4, 8) and ``perm_dim`` order is (2, 0, 1) then output 
tensor is of the shape (8, 2, 4). This transpose reflects changing the feature map layout from HWC to CHW.

Here is a list of all available permute functions:

.. table:: List of Available Permute Functions
   :align: center
   :widths: auto
   
   +---------------------------+------------------------------------+
   | **Function Name**         | **Details**                        |
   +===========================+====================================+
   | ``mli_krn_permute_sa8``   | All tensors data format: **sa8**   |
   +---------------------------+------------------------------------+
   | ``mli_krn_permute_fx8``   | All tensors data format: **fx8**   |
   +---------------------------+------------------------------------+
   | ``mli_krn_permute_fx16``  | All tensors data format: **fx16**  |
   +---------------------------+------------------------------------+
..

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the used version of the kernel.

 - Shape of ``out`` tensor must reflect already permuted  shape of ``in`` tensor.

 - ``in`` and ``out`` tensors must not point to overlapped memory regions.

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

 - Only first N (equal to ``rank`` of ``in`` tensor) values in permutation order array are considered 
   by kernel. All of them must be unique, nonnegative and less than the ``rank`` of the ``in`` tensor.

Result
^^^^^^

These functions modify:

 - Memory pointed by ``out.data.mem`` field.  
 - ``el_params`` field of ``out`` tensor. 

It is assumed that all the other fields and structures are properly populated 
to be used in calculations and are not modified by the kernel.

For **fx8** and **fx16** ``el_params`` field of ``in`` tensor is copyed to the ``out`` tensor. The same is applyed for
**sa8** versions of kernel in case of per-tensor quantization (``in.el_params.sa.dim`` < 0)

For **sa8** versions of kernel, and in case of per-axis quantization, the ``el_params`` field of the 
``out`` tensor is filled by the kernel using the quantization parameters of the ``in`` tensor. 
The following fields are affected:

    - ``out.el_params.sa.zero_point.mem.pi16`` and the related capacity field

    - ``out.el_params.sa.scale.mem.pi16`` and the related capacity field

    - ``out.el_params.sa.scale_frac_bits.mem.pi8`` and the related capacity field

Depending on the state of the preceding pointer fields, ensure that you choose only one of the following options to 
initialize all the fields in a consistent way:

    - If you initialize the pointers with a ``nullptr``, then the corresponding fields from the ``in`` tensor 
      are copied to the ``out`` tensor. No copy of quantization parameters itself is performed.

    - If you initialize the pointers with the corresponding fields from the ``in`` tensor, 
      then no action is applied.

    - If you initialize the pointers and capacity fields with pre-allocated memory and its capacity,
      then a copy of quantization parameters itself is performed. Capacity of allocated memory must 
      be big enough to keep related data from input tensor.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.



