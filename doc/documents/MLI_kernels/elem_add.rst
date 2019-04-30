.. _elmwise_add:

Element-wise Add
~~~~~~~~~~~~~~~~

This kernel adds two tensors of the same shape element-wise and
stores results to the output tensor saving the shape of inputs:

.. math:: y_{i} = {x1}_{i} + {x2}_{i}

..

It supports simple broadcasting of single value (scalar tensor see
:ref:`mli_tns_struct`) on general tensor. One of the operands can be a
scalar:

.. math:: y_{i} = x_{i} + x_{\text{scalar}}

..

The elements of input tensors must be of the same type and with the
same element parameters. Output tensor holds the same element and
shape parameters as input tensors (for example, a scalar tensor).

If the result of an operation is out of containers' range, it is
saturated to the container’s limit.

Kernel can perform in-place computation (output and input can point
to exactly the same memory but without shift). It can affect
performance for some platforms.

.. _function-configuration-structure-12:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No configuration structure for the kernel is required. All necessary
information is provided by tensors.

.. _api-8:

Kernel Interface
^^^^^^^^^^^^^^^^^

Prototype
'''''''''

.. code:: c                          
                                     
   mli_status mli_krn_eltwise_add_fx ( 
      const mli_tensor *in1,              
      const mli_tensor *in2,              
      mli_tensor *out);                   
..

Parameters 
''''''''''

.. table:: Kernel Interface Parameters
   :widths: 20,130
   
   +-----------------------+-----------------------+
   | **Parameters**        | **Description**       |
   +=======================+=======================+
   |                       |                       |
   | ``in1``               | [IN] Pointer to the   |
   |                       | first input tensor    |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``in2``               | [IN] Pointer to the   |
   |                       | second input tensor   |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``out``               | [OUT] Pointer to the  |
   |                       | output tensor. Result |
   |                       | is stored here        |
   +-----------------------+-----------------------+

.. _kernel-specializations-8:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

.. table:: Non-Specialized Functions
   :widths: 20,130
   
   +------------------------------+--------------------------------------+
   | **Function**                 | **Description**                      |
   +==============================+======================================+
   | ``mli_krn_eltwise_add_fx8``  | General function; 8bit FX elements;  |
   +------------------------------+--------------------------------------+
   | ``mli_krn_eltwise_add_fx16`` | General function; 16bit FX elements; |
   +------------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-8:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that you satisfy the following conditions before applying the function:

-  Input, tensors must be valid (see :ref:`mli_tns_struct`).

-  One of the inputs can be a valid scalar (see :ref:`mli_tns_struct`).

-  If both the inputs are tensors, the shape and rank of both of them must be equal.

-  Before processing, the output tensor must contain a valid pointer to a buffer, with sufficient capacity enough for storing the result. Other fields are filled by kernel (shape, rank and element specific parameters).