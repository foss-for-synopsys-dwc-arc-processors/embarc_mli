Common Group
------------

   Common group provides operations for common machine learning
   processing and other mathematical and statistic calculations.

Fully Connected
~~~~~~~~~~~~~~~

.. _f_fully_connected:
.. figure:: ../pic/images/image110.png

   Fully Connected Layer Schematic Representation

   This kernel implements fully connected layer, also usually referred
   to as the inner product or dense layer (see :numref:`f_fully_connected`).

   In :numref:`f_fully_connected`, *N* is the total number of elements in the input and M
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

Long Short Term Memory (LSTM) Cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _f_lstm:
.. figure:: ../pic/images/image119.png

   Long Short Term Memory Schematic Representation

   This kernel implements the default non-peephole implementation of
   Long short term memory cell (see :numref:`f_lstm`).

   The LSTM operation is described by the following formulas:

.. math::

   {i_{t} = sigm\left( x_{t}W_{\text{xi}} + h_{t - 1}W_{\text{hi}} + b_{i} \right)} 
..
  
.. math::
  
   {f_{t} = sigm\left( x_{t}W_{\text{xf}} + h_{t - 1}W_{\text{hf}} + b_{f} \right)}
..   

.. math::

   {o_{t} = sigm\left( x_{t}W_{\text{xo}} + h_{t - 1}W_{\text{ho}} + b_{o} \right)}
..

.. math::
   
   {g_{t} = \tanh\left( x_{t}W_{\text{xg}} + h_{t - 1}W_{\text{hg}} + b_{g} \right)}
..

.. math::
   
   {C_{t} = g_{t}*i_{t} + f_{t}*C_{t - 1}}
..

.. math::
   
   {h_{t} = o_{t}\ *o\_ act(C_{t})}

..
   
Where:

   :math:`\ x_{t}\ ` - frame :math:`t` in input sequence.

   :math:`\ h_{t}\ ` - cell output for frame :math:`t` in input
   sequence.

   :math:`i_{t}\ ,\ f_{t}\ ,\ o_{t}` – Input, forget, output gate
   subtensors for frame :math:`t` in input sequence.

   :math:`\ g_{t}\ ` - New cell candidates for frame :math:`t` in input
   sequence.

   :math:`\ C_{t}\ ` - Cell state for frame :math:`t` in input sequence.

   :math:`W_{**}\ ` - weights for appropriate input subtensor.

   :math:`b_{*}\ ` - bias for appropriate input subtensor.

   *sigm* , *tanh* - sigmoid and hyperbolic tangent
   activation functions.

   :math:`o\_ act` – output activation function.

   In :numref:`f_lstm`, *N* is the total number of elements in the input and M
   is the total number of elements in the cell output.

   Kernel supports various types of output activation (:math:`o\_ act`
   in the formula above):

   -  **Hyperbolic tangent**: Uses TanH kernel of the library (see :ref:`tanh`).
      Number of fractional bits for output tensor is the same as that for
      tensors processed by TanH activation.

   -  **Sigmoid**: Uses Sigmoid kernel of the library (see :ref:`sigmoid`). Number
      of fractional bits for output tensor is the same as that for tensors
      processed by Sigmoid activation.

   -  **No Activation**: Passes data without modification.

..

   The kernel takes 7 tensors including input, weights, cell,
   intermediate tensor from configuration structure and others (for full
   list, see :ref:`api_lstm`). It modifies only output tensor, cell tensors, and
   intermediate tensor in processing.

   Weights for cell is a single three-dimensional tensor of shape [4, *M*,
   *M+N*]. Ensure that bias is of shape [4, M]. It represents stacking
   of all weights sub tensors into one tensor in order (I, g, f, o):

.. math::

   \begin{bmatrix}
   \begin{matrix}
   W_{\text{xi}} \\
   W_{\text{xg}} \\
   \begin{matrix}
   W_{\text{xf}} \\
   W_{\text{xo}} \\
   \end{matrix} \\
   \end{matrix} & \begin{matrix}
   W_{\text{hi}} \\
   W_{\text{hg}} \\
   \begin{matrix}
   W_{\text{hf}} \\
   W_{\text{ho}} \\
   \end{matrix} \\
   \end{matrix} \\
   \end{bmatrix}\text{ }

..
   
   The first [M, *M+N]* sub-tensor of weights is applied to the input
   gate, the second, tor new cell candidates, the third, to the forget
   gate, and the last, to the output gate.

.. note::
   -  Ensure that you keep the same 
      order of sub-tensors for bias 
      tensor. For more information  
      about kernel parameters       
      requirements see :ref:`cond_lstm`.      
                                    
   -  Ensure that the configuration 
      structure (see :ref:`fn_conf_lstm`) also 
      contains the pointer to       
      tensor, which is used by      
      kernel as intermediate result 
      tensor. Kernel modifies the   
      memory pointed to by the data,
      shape, rank, element type and 
      element parameters fields of  
      this tensor.                  
                                    
   -  Ensure that the capacity of   
      the intermediate tensor is    
      enough to store M*4 elements  
      of input tensor type          

..

   Kernel supports three modes of input processing:

   -  **One-to-one**

      -  Processes the input tensor as a single input frame

      -  Ignores the shape of input tensor, and considers only the total
         number of elements

      -  Performs single step to produce one-dimensional output tensor of
         shape [*M*]

      -  Updates the memory pointed to by cell tensor, but does not modify
         the tensor’s fields

   -  **Batch-to-batch**

      -  Processes the input tensor as a sequence of frames to produce a
         sequence of outputs of the same size

      -  Considers first dimension of input tensor as sequence size
         (``batch_size``), and considers only the total number of elements
         for the rest of the dimensions

      -  Performs ``batch_size`` steps to produce two-dimensional output tensor
         of shape [``batch_size``, *M*]

      -  Updates the memory pointed to by cell tensor, but does not modify
         the tensor’s fields

   -  **Batch-to-last**

      -  Processes the input tensor as a sequence of frames to produce a
         single (last in the sequence) output

      -  Same as Batch-to-batch mode except that outputs tensor has a shape
         [*M*] whose values are the same as those for the last sub
         tensor in batch-to-batch mode

..

   Dense part of calculations uses intermediate tensor for result, and
   consequently output and previous output tensors might use the same
   memory if it is acceptable to rewrite previous output data.

.. note::
   Ensure that you allocate memory
   for the rest of the tensors    
   (including intermediate results
   tensor) without overlaps.      
   Otherwise the behavior is      
   undefined.                     

.. _fn_conf_lstm:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Definition**        | typedef struct {                              |
|                       |    mli_rnn_mode mode;                         |
|                       |    mli_rnn_out_activation  act;               |
|                       |    mli_tensor *ir_tsr;                        |
|                       |  } mli_rnn_cell_cfg;                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
| **Fields**            | ``mode``              | LSTM processing mode  |
|                       |                       | (enumeration)         |
+-----------------------+-----------------------+-----------------------+
|                       | ``act``               | LSTM output           |
|                       |                       | activation type       |
|                       |                       | (enumeration)         |
+-----------------------+-----------------------+-----------------------+
|                       | ``ir_tsr``            | Pointer to tensor for |
|                       |                       | holding intermediate  |
|                       |                       | results. Tensor must  |
|                       |                       | contain valid data    |
|                       |                       | and capacity fields.  |
|                       |                       | Field is modified by  |
|                       |                       | kernels.              |
+-----------------------+-----------------------+-----------------------+

\
  
.. _mli_rnn_mode_val_desc:
.. table:: mli_rnn_mode Values Description
   :widths: auto
   
   +-----------------------------------+-----------------------------------+
   | **Value**                         | **Field Description**             |
   +===================================+===================================+
   | ``RNN_ONE_TO_ONE``                | Process input tensor as a single  |
   |                                   | input frame .                     |
   +-----------------------------------+-----------------------------------+
   | ``RNN_BATCH_TO_BATCH``            | Process input tensor as a         |
   |                                   | sequence of frames to produce a   |
   |                                   | sequence of outputs .             |
   +-----------------------------------+-----------------------------------+
   | ``RNN_BATCH_TO_LAST``             | Process input tensor as a         |
   |                                   | sequence of frames to produce     |
   |                                   | single (last) outputs.            |
   +-----------------------------------+-----------------------------------+

\

.. _mli_rnn_out_activation_val_desc:
.. table:: mli_rnn_out_activation Values Description
   :widths: auto
   
   +------------------+-----------------------------------------+
   | **Value**        | **Field Description**                   |
   +==================+=========================================+
   | ``RNN_ACT_TANH`` | Hyperbolic tangent activation function. |
   +------------------+-----------------------------------------+
   | ``RNN_ACT_SIGM`` | Logistic (sigmoid) activation function. |
   +------------------+-----------------------------------------+
   | ``RNN_ACT_NONE`` | No activation.                          |
   +------------------+-----------------------------------------+

\

.. _api_lstm:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_lstm_cell_<data_type>      |
|                       | [_specialization](                            |
|                       |    const mli_tensor *in,                      |
|                       |    const mli_tensor *prev_out,                |
|                       |    const mli_tensor *weights,                 |
|                       |    const mli_tensor *bias,                    |
|                       |    const mli_lstm_cell_cfg *cfg,              |
|                       |    mli_tensor *cell,                          |
|                       |    mli_tensor *out);                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       | ``prev_out``          | [IN] Pointer to       |
|                       |                       | previous output       |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       | ``weights``           | [IN] Pointer to       |
|                       |                       | weights tensor        |
+-----------------------+-----------------------+-----------------------+
|                       | ``bias``              | [IN] Pointer to       |
|                       |                       | biases tensor         |
+-----------------------+-----------------------+-----------------------+
|                       | ``cfg``               | [IN/OUT] Pointer to   |
|                       |                       | configuration         |
|                       |                       | structure             |
+-----------------------+-----------------------+-----------------------+
|                       | ``cell``              | [IN/OUT] Pointer to   |
|                       |                       | cell state tensor     |
+-----------------------+-----------------------+-----------------------+
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor.        |
+-----------------------+-----------------------+-----------------------+

\

.. _kernel-specializations-1:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+-----------------------------------+
| **Function**                      | **Description**                   |
+===================================+===================================+
| ``mli_krn_lstm_cell_fx8``         | General function; 8bit FX         |
|                                   | elements;                         |
+-----------------------------------+-----------------------------------+
| ``mli_krn_lstm_cell_fx16``        | General function; 16bit FX        |
|                                   | elements;                         |
+-----------------------------------+-----------------------------------+
| ``mli_krn_lstm_cell_fx8w16d``     | General function; FX tensors      |
|                                   | (8bit weights and biases, 16 bit  |
|                                   | input, state, cell, output and    |
|                                   | intermediate data);               |
+-----------------------------------+-----------------------------------+

.. _cond_lstm:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, Weights, Bias, cell, and Previous output tensors must be valid
      (see :ref:`mli_tns_struct`)

   -  Weights must be a three-dimensional tensor of shape [4, M, N+M]

   -  Bias must be a two-dimensional tensor of shape [4, M]

   -  Cell must be a one-dimensional tensor of shape [M]

   -  Previous output must be a one-dimensional tensor of shape [M]

   -  Element type of Weights and Bias tensors must be the same

   -  Element type of Input, Cell and Previous output tensors must be the
      same

   -  The Input tensor has the following restrictions:

      -  For ``RNN_ONE_TO_ONE`` mode, the total number of input and previous
         output tensors (N+M) must be equal to the last dimension of
         Weights tensor

      -  For ``RNN_BATCH_TO_BATCH`` and ``RNN_BATCH_TO_LAST`` modes, the first
         dimension of input reflects sequence length (batch size) while for
         the rest of the input tensor dimensions, the same rules apply as
         for ``RNN_ONE_TO_ONE`` mode

   -  The output tensor has the following restrictions:

      -  It must contain a valid pointer to a buffer with sufficient
         capacity for storing the result (to keep M elements for
         ``RNN_ONE_TO_ONE`` and ``RNN_BATCH_TO_LAST`` modes, and M*batch_size
         elements for ``RNN_BATCH_TO_BATCH`` mode)

      -  If ``RNN_ACT_NONE`` is used as output activation, output tensor must
         contain a valid element parameter (``el_params.fx.frac_bits``) and it
         must be the same as for the previous output tensor

      -  Before processing, the output tensor does not have to contain a
         valid shape, rank, or element type. These are filled by function
         according to inputs and kernel processing mode. If ``RNN_ACT_NONE`` is
         not used, the same applies to element parameter
         (``el_params.fx.frac_bits``)

      -  Before processing, intermediate result tensor in config structure
         must contain a valid pointer to a buffer with sufficient capacity
         for the result (4*M elements of input type)

Basic RNN Cell
~~~~~~~~~~~~~~

.. _f_basic_rnn_cell:
.. figure:: ../pic/images/image139.png 

   Basic RNN Cell Schematic Representation

   This kernel implements the basic recurrent cell without memory state
   (see :numref:`f_basic_rnn_cell`).

   In :numref:`f_basic_rnn_cell`, *N* is the total number of elements in the input and M
   is the total number of elements in the cell output.

   Basic RNN operation is described by the following formula:

.. math:: h_{t} = f(x_{t}W_{x} + h_{t - 1}W_{h} + b)

..

   Where:

   :math:`\ x_{t}\ ` - frame :math:`t` in input sequence.

   :math:`\ h_{t}\ ` - cell output for frame :math:`t` in input
   sequence.

   :math:`W_{*}\ ` - weights for appropriate input subtensor.

   :math:`b_{*}\ ` - bias for appropriate input subtensor.

   :math:`f()` - output activation function.

   Kernel supports following types of output activation (:math:`f()` in
   the formula above) :

   -  Hyperbolic tangent. Uses TanH kernel of the library (see :ref:`tanh`).

   -  Sigmoid. Uses Sigmoid kernel of the library (see :ref:`sigmoid`)

   -  No Activation. Passes data without modification

..

   Kernel modifies only output tensors and intermediate tensor from
   configuration structure in processing. For a full list of parameters
   see :ref:`api_brnn`.

   Kernel supports three modes of input processing

   -  **One-to-one**

      -  Processes the input tensor as a single input frame

      -  Ignores the shape of input tensor, and only considers the total
         number of elements

      -  Performs single step to produce a one-dimensional output tensor of
         shape [M]

   -  **Batch-to-batch**

      -  Processes the input tensor as a sequence of frames to produce a
         sequence of outputs of the same size

      -  Considers the first dimension of input tensor as sequence size
         (``batch_size``), and considers the total number of elements for the
         rest of the dimensions.

      -  Performs ``batch_size`` steps to produce 2 dimensional output tensor
         of shape [``batch_size``, M].

   -  **Batch-to-last**

      -  Processes the input tensor as a sequence of frames to produce a
         single (last in the sequence) output

      -  Same as batch-to-batch mode except that outputs tensor has a shape
         [M] whose values are the same as those for the last sub tensor in
         batch-to-batch mode

..

   Weights for a cell is a single 2-dimensionl tensor of shape [*M*,
   *M+N*], an Bias is of shape [M]. It represents the stacking of 2
   weights sub-tensors into one tensor in the following order:

.. math::

   \begin{bmatrix}
   W_{x} & W_{h} \\
   \end{bmatrix}\text{ }

..
   
   To support user-specific complex recurrent cells beside LSTM, basic
   RNN cell kernel in One-to-One mode can work with matrixes with
   stacked weights to produce stacked output tensor.

   For example, if weights tensor is 3-dimensionl tensor of shape [*L*,
   *M*, *M+N*], and Bias of shape [*L, M*], the output tensor is of
   shape [*L*, *M*].

   In batch-to-last mode, configuration structure also contains pointer
   to the tensor that is used by kernel as intermediate result tensor.
   Kernel modifies the memory pointed to by data, shape, rank, element
   type and element parameters fields of this tensor. Ensure that the
   capacity of the intermediate tensor is enough to store the output for
   one step of kernel (M or L*M elements for stacked weights matrix).

   For the other modes (one-to-one or batch-to-batch) kernel does not
   use the intermediate result tensor and this field might not be
   initialized. For more information about configuration structure see
   :ref:`fn_conf_lstm`.
   
.. note::
   Ensure that you allocate memory 
   for all tensors (including      
   intermediate results tensor)    
   without overlaps.               
                                   
   The only exception is           
   batch-to-last mode due to its   
   usage of intermediate tensor. In
   this case, the output and the previous   
   output tensors might use the same 
   memory if it is acceptable to   
   rewrite previous output data.   

.. _fn_conf_brnn:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Basic RNN cell kernel shares configuration structure with LSTM cell.
   For more information see :ref:`fn_conf_lstm`.

.. _api_brnn:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | mli_status mli_krn_basic_rnn_cell_<data_type> |
|                       | [_specialization](                            |
|                       |    const mli_tensor *in,                      |
|                       |    const mli_tensor *prev_out,                |
|                       |    const mli_tensor *weights,                 |
|                       |    const mli_tensor *bias,                    |
|                       |    const mli_rnn_cell_cfg *cfg,               |
|                       |    mli_tensor *out);                          |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``prev_out``          | [IN] Pointer to       |
|                       |                       | previous output       |
|                       |                       | tensor                |
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
|                       | ``cfg``               | [IN/OUT] Pointer to   |
|                       |                       | configuration         |
|                       |                       | structure             |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
|                       | ``out``               | [OUT] Pointer to      |
|                       |                       | output tensor. Result |
|                       |                       | is stored here        |
+-----------------------+-----------------------+-----------------------+

.. _kernel-specializations-2:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------------+-----------------------------------+
| **Function**                          | **Description**                   |
+=======================================+===================================+
| ``mli_krn_basic_rnn_cell_fx8``        | General function; 8bit FX         |
|                                       | elements;                         |
+---------------------------------------+-----------------------------------+
| ``mli_krn_basic_rnn_cell_fx16``       | General function; 16bit FX        |
|                                       | elements;                         |
+---------------------------------------+-----------------------------------+
| ``mli_krn_basic_rnn_cell_fx8w16d``    | General function; FX tensors      |
|                                       | (8bit weights and biases, 16 bit  |
|                                       | input, state, cell, output and    |
|                                       | intermediate data);               |
+---------------------------------------+-----------------------------------+

.. _conditions-for-applying-the-kernel-2:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Ensure that you satisfy the following conditions before applying the
   function:

   -  Input, Weights, Bias, and Previous output tensors must be valid (see
      :ref:`mli_tns_struct`).

   -  Weights is a two-dimensional tensor of shape [M, N+M]. But In
      ``RNN_ONE_TO_ONE`` mode, the weights tensor is of shape [L, M, N+M] to
      produce an output tensor of shape [L, M].

   -  Bias is a one-dimensional tensor of shape [M]. But In ``RNN_ONE_TO_ONE``
      mode, bias tensor is of shape [L, M] to produce an output tensor
      of shape [L, M].

   -  Previous output must be a one-dimensional tensor of shape [M]

   -  Element type of Weights and Bias tensors must be the same.

   -  Element type of Input, Previous output tensors must be the same.

   -  The input tensor has the following restrictions:

      -  For ``RNN_ONE_TO_ONE`` mode, the total number of input and previous
         output tensors (N+M) must be equal to the last dimension of
         Weights tensor.

      -  For ``RNN_BATCH_TO_BATCH`` and ``RNN_BATCH_TO_LAST`` modes, first
         dimension of input reflects sequence length (batch size) while for
         the rest of the input tensor dimensions the same rules apply as
         those for the ``RNN_ONE_TO_ONE`` mode.

   -  The output tensor has the following restrictions:
 
      -  It must contain a valid pointer to a buffer with sufficient
         capacity for storing the result (to keep *M* or *L*M* elements for
         RNN_ONE_TO_ONE and RNN_BATCH_TO_LAST modes, and *M*\ \*batch_size
         elements for RNN_BATCH_TO_BATCH mode)

      -  If ``RNN_ACT_NONE`` is used as output activation, output tensor must
         contain a valid element parameter (el_params.fx.frac_bits) and it
         must be the same as that for the previous output tensor.

      -  Before processing, the output tensor does not have to contain a
         valid shape, rank and element type. These are filled by function
         according to inputs, and kernel processing mode. If RNN_ACT_NONE
         is not used, the same rule applies for element parameter
         (``el_params.fx.frac_bits``).

   -  The intermediate result tensor in config structure has the following
      restrictions:

      -  For ``RNN_BATCH_TO_LAST`` mode, it must contain a valid pointer to a
         buffer with sufficient capacity for storing the result (M elements
         of input type).

      -  In other cases, this tensor is not used and might be used to hold
         any data.
