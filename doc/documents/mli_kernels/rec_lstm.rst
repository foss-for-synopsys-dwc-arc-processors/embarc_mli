Basic Long Short Term Memory (LSTM) Cell Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel implements the basic non-peephole Long Short-Term Memory (LSTM) cell 
(see `Long Short-term Memory <https://en.wikipedia.org/wiki/Long_short-term_memory>`_ 
for more details), as shown in Figure :ref:`f_lstm_schematic`. 
 
.. _f_lstm_schematic:
.. figure:: ../images/lstm_schematic.png
   :align: center
 
   Long Short Term Memory Schematic Representation
..

.. _x_lstm:

The LSTM Operation
^^^^^^^^^^^^^^^^^^

The LSTM operation is described by the following formulas:


.. math::

   {i_{t}} &= {sigm(x_{t}W_{\text{xi}} + h_{t - 1}W_{\text{hi}} + b_{i})}
   
   {f_{t}} &= {sigm(x_{t}W_{\text{xf}} + h_{t - 1}W_{\text{hf}} + b_{f})}
      
   {o_{t}} &= {sigm(x_{t}W_{\text{xo}} + h_{t - 1}W_{\text{ho}} + b_{o})}
   
   {g_{t}} &= {tanh(x_{t}W_{\text{xg}} + h_{t - 1}W_{\text{hg}} + b_{g})}
   
   {C_{t}} &= {g_{t}*i_{t} + f_{t}*C_{t - 1}}
   
   {h_{t}} &= {o_{t}\ * tanh(C_{t})}
..

Where:

   :math:`\ x_{t}\ ` *- frame* :math:`t` *in input sequence.*

   :math:`\ h_{t}\ ` *- cell output for frame* :math:`t` *in input
   sequence.*

   :math:`i_{t}\ ,\ f_{t}\ ,\ o_{t}` *– Input, forget, output gate
   subtensors for frame* :math:`t` *in input sequence.*

   :math:`\ g_{t}\ ` *- New cell candidates for frame* :math:`t` *in
   input sequence.*

   :math:`\ C_{t}\ ` *- Cell state for frame* :math:`t` *in input
   sequence.*

   :math:`W_{**}\ ` *- weights for appropriate input subtensor.*

   :math:`b_{*}\ ` *- bias for appropriate input subtensor.*

   :math:`sigm` , :math:`tanh` *- sigmoid and hyperbolic tangent
   activation functions.*


In the Figure :ref:`f_lstm_schematic`, N is the total number of 
elements in the input and M is the total number of elements in the cell output.

Kernels which implement an LSTM cell have the following prototype:

.. code:: c

   mli_status mli_krn_lstm_cell_<data_format>(
      const mli_tensor *in,
      const mli_tensor *prev_out,
      const mli_tensor *weights_in,
      const mli_tensor *weights_out,
      const mli_tensor *bias,
      const mli_rnn_cell_cfg *cfg,
      mli_tensor *cell,
      mli_tensor *out);
..

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function parameters 
are shown in the following table:

.. table:: LSTM Function parameters
   :align: center
   :widths: auto 
   
   +------------------+-------------------------+-----------------------------------------------------------------+
   | **Parameter**    | **Type**                | **Description**                                                 |
   +==================+=========================+=================================================================+
   | ``in``           | ``mli_tensor *``        | [IN] Pointer to constant input tensor.                          |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``prev_out``     | ``mli_tensor *``        | [IN] Pointer to constant previous output tensor.                |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``weights_in``   | ``mli_tensor *``        | [IN] Pointer to constant weights tensor for LSTM input.         |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``weights_out``  | ``mli_tensor *``        | [IN] Pointer to constant weights tensor for LSTM output.        |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``bias``         | ``mli_tensor *``        | [IN] Pointer to constant bias tensor.                           |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``cfg``          | ``mli_rnn_cell_cfg *``  | [IN/OUT]   Pointer to RNN cell parameters structure.            |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``cell``         | ``mli_tensor *``        | [IN/OUT] Pointer to cell tensor. Is modified during execution.  |
   +------------------+-------------------------+-----------------------------------------------------------------+
   | ``out``          | ``mli_tensor *``        | [OUT] Pointer to output tensor. Result is stored here.          |
   +------------------+-------------------------+-----------------------------------------------------------------+
..

Fields of ``mli_rnn_cell_cfg`` structure are described in the Table :ref:`t_mli_rnn_cell_cfg_desc`.

Weights for the cell consist of three tensors:

 - ``weights_in``: a three-dimensional tensor of shape (4, N, M) where N is a number of elements 
   in input tensor, and M is a number of cell elements (equal to number of elements in cell state 
   and output tensor). It represents stacking of weights from :ref:`x_lstm` in the order 
   (I, g, f,o):

.. math::

   \begin{bmatrix}
   W_{\text{xi}} & W_{\text{xg}} & \begin{matrix}
   W_{\text{xf}} & W_{\text{xo}} \\
   \end{matrix} \\
   \end{bmatrix}
..

 - ``weights_out``: a three-dimensional tensor of shape (4, M, M) where M is a number of cell 
   elements (weights which involved into a single dot    product series are stored column-wise, 
   that is, with M stride in memory). It represents stacking of weights from :ref:`x_lstm` in
   order (I, g, f, o):

.. math::

   \begin{bmatrix}
   W_{\text{hi}} & W_{\text{hg}} & \begin{matrix}
   W_{\text{hf}} & W_{\text{ho}} \\
   \end{matrix} \\
   \end{bmatrix}
..

 - ``bias`` tensor of shape (4, M) keeps subtensors in the same order:

.. math::

   \begin{bmatrix}
   b_{i} & b_{g} & \begin{matrix}
   b_{f} & b_{o} \\
   \end{matrix} \\
   \end{bmatrix} 
..
   
This kernel implies sequential processing of the set of input vectors that is passed by input tensor 
of shape (batch_size, N) where N is the length of the single frame :math:`x_{t}`. Both directions 
of processing (forward and backward) are supported and defined by cfg structure. The Kernel can output 
a pack of results at each step of processing, or it can output the result vector only for the last 
step in the sequence.
 
Dense part of calculations uses scratch data from configuration structure for results, and consequently 
output and previous output tensors might use the same memory if it is acceptable to rewrite previous output 
data. Ensure that you allocate memory for the rest of the tensors and for scratch data from cfg structure 
without overlaps. Otherwise the behavior is undefined.

Here is a list of all available LSTM cell functions:

.. table:: List of Available LTSM Cell Functions
   :align: center
   :widths: auto 
   
   +-------------------------------------+-------------------------------------------+
   | **Function Name**                   | **Details**                               |
   +=====================================+===========================================+
   | ``mli_krn_lstm_cell_sa8_sa8_sa32``  || In/out/cell/weights data format: **sa8** |
   |                                     || Bias data format: **sa32**               |
   +-------------------------------------+-------------------------------------------+
   | ``mli_krn_lstm_cell_fx16``          || All tensors data format: **fx16**        |
   +-------------------------------------+-------------------------------------------+
   | ``mli_krn_lstm_cell_fx16_fx8_fx8``  || In/out/cell data format: **fx16**        |
   |                                     || weights/Bias data format: **fx8**        |
   +-------------------------------------+-------------------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in``, ``prev_out``, ``weights_in``, ``weights_out``, ``bias``, and ``cell`` tensors must be valid.

 - ``in`` must be a tensor of shape (batch_size, N) where batch_size is a number of input frames for sequential 
   processing by LSTM cell.

 - ``weights_in`` must be a three-dimensional tensor of shape (4, N, M).
 
 - ``weights_out`` must be a three-dimensional tensor of shape (4, M, M).
 
 - ``bias`` must be a two-dimensional tensor of shape (4, M).
 
 - ``cell`` must be a one-dimensional tensor of shape (M).
 
 - ``prev_out`` must be a one-dimensional tensor of shape (M).
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity and valid el_params union. 
   Other fields of the structure do not have to contain valid data and are filled by the function.
   
 - ``in`` and ``cfg->scratch_data`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` must contain a valid pointer to a buffer with sufficient capacity for storing the result (to keep M 
   elements if LSTM cell is configured with RNN_OUT_LAST or to keep M*batch_size elements if LSTM cell is configured 
   with RNN_OUT_ALL). Other fields of the structure do not have to contain valid data and are filled by the function.
   
 - Before processing, scratch_data field in config structure must contain a valid pointer to a buffer with enough 
   capacity for the result (4*M elements of input type). The ``scratch_capacity`` field must reflect the available size of 
   this memory in bytes properly (see Table :ref:`t_mli_rnn_cell_cfg_desc`). 
   
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following conditions before calling the function: 

 - ``in``, ``prev_out`` and ``cell`` tensor must be quantized on the tensor level. It implies that each tensor contains a 
   single scale factor and a single zero offset.
   
 - ``weights_in``, ``weights_out`` and ``bias`` tensors must be symmetric and quantized per first dimension (number of 
   sub-tensors equal to 4). It implies that each tensor contains separate scale point for each sub-tensor. All tensors 
   contain single zero offset equal to 0.
   
 - Scale factors of bias tensor must be equal to the multiplication of input scale factor broadcasted on ``weights_in`` 
   array of scale factors.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

These kernels modify ``out`` tensor, ``cell`` tensors, and memory pointed by ``scratch_data`` field of cfg structure.

