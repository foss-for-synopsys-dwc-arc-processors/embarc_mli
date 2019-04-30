.. _permute:
     
Permute
~~~~~~~

The kernel permutes dimensions of input tensor according to provided
order. In other words, it transposes input tensors.

The new order of dimensions is given by ``perm_dim`` array of kernel
configuration structure. Output dimension ``idx`` corresponds to the
dimension of input tensor with ``perm_dim[idx]``. Tensor’s data is
reordered according to new shape.

For example, if input tensors have the shape [2, 4, 8] and ``perm_dim``
order is (2, 0, 1) then output tensor is of the shape [8, 2, 4]. This
transpose reflects changing the feature map layout from HWC to CHW.

.. caution::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.

.. _function-configuration-structure-17:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Definition
''''''''''

.. code:: c                    
                               
   typedef struct {              
      int perm_dim[MLI_MAX_RANK];
   }  mli_permute_cfg;           


Parameters
''''''''''

.. table:: Function Configuration Parameters
 
	+-----------------------+-----------------------+
	| **Field**             | **Description**       |
	+=======================+=======================+
	|                       |                       |
	| ``perm_dim``          | A permutation array.  |
	|                       | Dimensions order for  |
	|                       | output tensor.        |
	+-----------------------+-----------------------+

.. _api-13:

Kernel Interface
^^^^^^^^^^^^^^^^

Prototype
'''''''''

.. code:: c                     
                                
    mli_status mli_krn_permute_fx (
       const mli_tensor *in,       
       const mli_permute_cfg *cfg, 
       mli_tensor *out);           
..

Parameters
''''''''''

.. table:: Kernel Interface Parameters

	+-----------------------+-----------------------+
	| **Parameters**        | **Description**       |
	+-----------------------+-----------------------+
	|                       |                       |
	| ``in``                | [IN] Pointer to input |
	|                       | tensor                |
	+-----------------------+-----------------------+
	|                       |                       |
	| ``cfg``               | [IN] Pointer to       |
	|                       | configuration         |
	|                       | structure with        |
	|                       | permutation order     |
	+-----------------------+-----------------------+
	|                       |                       |
	| ``out``               | [OUT] Pointer to the  |
	|                       | output tensor. Result |
	|                       | is stored here        |
	+-----------------------+-----------------------+

.. _kernel-specializations-13:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

.. table:: Non-Specialized Functions
   :widths: 20,130
   
   +--------------------------+--------------------------------------+
   | **Function**             | **Description**                      |
   +==========================+======================================+
   | ``mli_krn_permute_fx8``  | General function; 8bit FX elements;  |
   +--------------------------+--------------------------------------+
   | ``mli_krn_permute_fx16`` | General function; 16bit FX elements; |
   +--------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-13:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that you satisfy the following conditions before applying the
function:

-  Input, tensors must be valid (see :ref:`mli_tns_struct`)

-  Only first N=rank_of_input_tensor values in permutation order array
   are considered by kernel. All of them must be unique, non-negative
   and less then rank of the input tensor

-  Before processing, output tensor must contain a valid pointer to a
   buffer with sufficient capacity enough for storing the result
   (that is, the total amount of elements in input tensor). Other
   fields are filled by kernel (shape, rank and element specific
   parameters)

-  Buffers of input and output tensors must point to different
   not-overlapped memory regions.