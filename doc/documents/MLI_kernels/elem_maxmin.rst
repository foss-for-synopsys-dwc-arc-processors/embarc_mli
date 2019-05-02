.. _elmwise_maxmin:

Element-wise MAX/MIN
~~~~~~~~~~~~~~~~~~~~

Find element-wise maximum / minimum of inputs operands and store
results to the output tensor:

.. math::

   {y_{i} = \text{MAX}\left( {x^1}_{i}\ \ ,\ \ \ {x^2}_{i} \right)}
   
   {y_{i} = \text{MIN}\left( {x^1}_{i}\ \ ,\ \ \ {x^2}_{i} \right)\ }

..
   
Simple broadcasting of single value (scalar tensor see :ref:`mli_tns_struct`
) on general tensor also supported. The only one operand might
be scalar but does not matter which of them:

.. math::

   {y_{i} = \text{MAX}\left( x_{\text{scalar}}\ \ ,\ \ \ x_{i} \right)}
   
   {y_{i} = \text{MIN}\left( x_{\text{scalar}}\ \ ,\ \ \ x_{i} \right)}

..
   
Elements of input tensors must be of the same type and with the same
element parameters. Output tensor holds the same element and
shape parameters as the input tensors (for example, a scalar tensor).

This kernel can perform in-place computation (output and input can point
to exactly the same memory but without shift). It can affect
performance for some platforms.

.. _function-configuration-structure-14:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No configuration structure for the kernel is required. All necessary
information is provided by tensors.

.. _api-10:

Kernel Interface
^^^^^^^^^^^^^^^^

Prototype
'''''''''

.. code:: c                               
                                         
   mli_status mli_krn_eltwise_[min/max]_fx (
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
   | ``out``               | [OUT] Pointer to      |
   |                       | output tensor. Result |
   |                       | is stored here        |
   +-----------------------+-----------------------+

.. _kernel-specializations-10:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

.. table:: Non-Specialized Functions
   :widths: 20,130
   
   +-----------------------------------+------------------------------------+
   | **Function**                      | **Description**                    |
   +===================================+====================================+
   | ``mli_krn_eltwise_max_fx8``       | General element-wise max function; |
   |                                   | 8bit FX elements;                  |
   +-----------------------------------+------------------------------------+
   | ``mli_krn_eltwise_max_fx16``      | General element-wise max function; |
   |                                   | 16bit FX elements;                 |
   +-----------------------------------+------------------------------------+
   | ``mli_krn_eltwise_min_fx8``       | General element-wise min function; |
   |                                   | 8bit FX elements;                  |
   +-----------------------------------+------------------------------------+
   | ``mli_krn_eltwise_min_fx16``      | General element-wise min function; |
   |                                   | 16bit FX elements;                 |
   +-----------------------------------+------------------------------------+

.. _conditions-for-applying-the-kernel-10:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that you satisfy the following conditions before applying the
function:

-  Input tensors must be valid (see :ref:`mli_tns_struct`)

-  One of the inputs can be a valid scalar (see :ref:`mli_tns_struct`).

-  If both the inputs are tensors, the shape and rank of both
   of them must be equal.

-  Before processing, the output tensor must contain a valid pointer to
   a buffer with sufficient capacity for storing the result.
   Other fields are filled by kernel (shape, rank, and element
   specific parameters)