.. _argmax_prot:

Argmax Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel returns the indexes of maximum values across the whole tensor, or for each slice 
across a dimension. 

Argmax functions have the following prototype:

.. code:: c

   mli_status mli_krn_argmax_<data_format>(
      const mli_tensor *in,
      const mli_argmax_cfg *cfg,	
      mli_tensor *out);	
..
   
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: Argmax Function Parameters
   :align: center
   :widths: auto
   
   +----------------+------------------------+----------------------------------------------+
   | **Parameter**  | **Type**               | **Description**                              |
   +================+========================+==============================================+
   | ``in``         | ``mli_tensor *``       | [IN] Pointer to constant input tensor.       |
   +----------------+------------------------+----------------------------------------------+
   | ``cfg``        | ``mli_argmax_cfg *``   | [IN] Pointer to argmax parameters structure. |
   +----------------+------------------------+----------------------------------------------+
   | ``out``        | ``mli_tensor *``       | [OUT] Pointer to output tensor.              |
   |                |                        | Result is stored here                        |
   +----------------+------------------------+----------------------------------------------+
..

   ``mli_argmax_cfg`` is defined as:
   
.. code:: c

   typedef struct {
        int32_t axis;
        int32_t topk;
   } mli_argmax_cfg;
..

.. _t_mli_argmax_cfg_desc:
.. table:: mli_argmax_cfg Structure Field Description
   :align: center
   :widths: auto
   
   +----------------+----------------+---------------------------------------------------------------------------+
   | **Field name** | **Type**       | **Description**                                                           |
   +================+================+===========================================================================+
   |                |                | An axis along which the function is computed. Axis corresponds to         |
   | ``axis``       | ``int32_t``    | index of tensor’s dimension starting from 0. For instance, having feature |
   |                |                | map in HWC layout, axis == 0 corresponds to H dimension. If axis < 0,     |
   |                |                | the function is applied to the whole tensor.                              |
   +----------------+----------------+---------------------------------------------------------------------------+
   | ``top_k``      | ``int32_t``    | Number of indexes per slice to be returned.                               |
   +----------------+----------------+---------------------------------------------------------------------------+
..

.. table:: List of Available Argmax Functions:
   :align: center
   :widths: auto
   
   +----------------------------+--------------------------------------+
   | **Function Name**          | **Details**                          |
   +============================+======================================+
   | ``mli_krn_argmax_sa8``     | ``in`` tensor data format: **sa8**   |
   +----------------------------+--------------------------------------+
   | ``mli_krn_argmax_fx16``    | ``in`` tensors data format: **fx16** |
   +----------------------------+--------------------------------------+
..   

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` tensor must be valid (see :ref:`mli_tnsr_struc`).
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain only
   
    - A valid pointer to a buffer with sufficient capacity. That is ``top_k*in.shape[axis]`` values
      of int32 type. 

    - A valid mem_stride field.
      
    - Other fields of the structure do not have to contain valid data and are filled by the function.

For **sa8** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following condition before calling the function:
 
 - ``in`` tensor must be quantized on the tensor level. This implies that the tensor 
   contains a single scale factor and a single zero offset.
   
Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

The Kernel modifies the output tensor which is transformed into two-dimensional tensor of shape 
``(dim_size, top_k]`` where ``dim_size`` is the size of dimension specified by the axis parameter in 
``mli_argmax_cfg`` structure, and ``top_k`` is the number of indexes per slice specified by the 
``topk`` parameter of the same structure. 

``el_type`` field of ``out`` tensor is set by the kernel to ``MLI_EL_SA_32`` and ``el_params`` field 
is configured to reflect fully integer values (zero_offset = 0,  scale = 1 and scale_frac_bits = 0). 
Values in output tensor are 32 bit indexes.  An Index represents the position of Nth 
(N<``top_k``) maximum value in the flattened slice across the defined dimension in the input tensor.