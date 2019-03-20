.. _fully_conn:

Fully Connected
~~~~~~~~~~~~~~~

.. image:: ../images/image110.png
   :align: center 
   :alt: Fully Connected Layer Schematic Representation

..

   This kernel implements fully connected layer, also usually referred
   to as the inner product or dense layer.

   *N* is the total number of elements in the input and M
   is the total number of neurons and is equal to output length. It
   performs dot product operation of input tensor with each row of
   weights matrix and adds biases to result:

.. math:: y_{i} = x_{j}W_{i,j} + b_{i}

..

   Where:

   :math:`\ x_{j}\ ` - :math:`j_{\text{th}}` value in input tensor.

   :math:`\ y_{i}\ ` - :math:`i_{\text{th}}` value in output tensor.

   :math:`W_{\text{ij}}\ ` - weight of :math:`j_{\text{th}}` input
   element for :math:`i_{\text{th}}` neuron.

   :math:`b_{j}\ ` - bias for :math:`i_{\text{th}}` neuron.

   Ensure that the weight for this kernel is a two-dimensional tensor
   (matrix) of shape [M, N], and Bias must be of shape [M]. Shape of
   input tensor is not considered and only total number of elements is
   considered. Kernel outputs a one-dimensional tensor of shape [M].

.. note::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.                  
      
.. _function-configuration-structure-4:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   No configuration structure for fully connected kernel is required.
   All necessary information is provided by tensors.

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_fully_connected_           |
|                       | <data_type>[_specialization](                 |
|                       |    const mli_tensor *in,                      |
|                       |    const mli_tensor *weights,                 |
|                       |    const mli_tensor  *bias,                   |
|                       |    mli_tensor *out);                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to Input |
|                       |                       | feature map tensor    |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``weights``           | [IN] Pointer to       |
|                       |                       | weights tensor        |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``bias``              | [IN] Pointer to       |
|                       |                       | biases tensor         |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-------------------------------------+-----------------------------------+
| **Function**                        | **Description**                   |
+=====================================+===================================+
| ``mli_krn_fully_connected_fx8``     | General function; 8bit FX         |
|                                     | elements;                         |
+-------------------------------------+-----------------------------------+
| ``mli_krn_fully_connected_fx16``    | General function; 16bit FX        |
|                                     | elements;                         |
+-------------------------------------+-----------------------------------+
| ``mli_krn_fully_connected_fx8w16d`` | General function; FX tensors      |
|                                     | (8bit weights and biases, 16 bit  |
|                                     | input and output);                |
+-------------------------------------+-----------------------------------+

.. _conditions_apply_kernel:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, Weights and Bias tensors must be valid (see 
      :ref:`mli_tns_struct`).

   -  Weights must be two-dimensional tensor

   -  Bias must be one-dimensional tensor. Its length must be equal to
      number of neurons (first dimension of weights tensor)

   -  Input tensor might be of any shape and rank

   -  The second dimension of weight tensor must be equal to length of
      input tensor (number of its elements)

   -  Before processing, the output tensor does not have to contain valid
      shape, rank, and element type fields. These are filled by the
      function

   -  Before processing, the output tensor must contain valid pointer to a
      buffer with sufficient capacity (enough for result storing). It
      also must contain a valid element parameter
      (``el_params.fx.frac_bits``)