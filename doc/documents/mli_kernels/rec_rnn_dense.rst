RNN Dense Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel implements a single basic fully connected (or dense) calculation 
typically used in the majority of RNN architectures:

.. math:: 

   y_{i} = b_{i} + \sum_{j}^{}{xa}_{j}*{Wa}_{i,j} + 
                 \ \sum_{j}^{}{xb}_{j}*{Wb}_{i,j} + 
		   \ldots\ \sum_{j}^{}{xn}_{j}*{Wn}_{i,j}
..

Where:

 -  :math:`{xa}_{j}`, :math:`{xb}_{j}`, :math:`{xn}_{j}` *–*
    :math:`j_{\text{th}}` *value in one of the input tensors. These input
    tensors might be current input, previous output, cell state or any other 
    tensor depending on RNN Cell architecture*

 -  :math:`{Wa}_{i,j}, :math:`{Wb}_{i,j}`, :math:`text{Wc}_{i,j}` *– weight
    of* :math:`j_{th}\ `\ *input element for*
    :math:`i_{th}` *neuron in one of input weights tensors. These
    weights tensors might be input-to-a-gate weights, output-to-a-gate
    weights or any other tensor depending on RNN Cell architecture*

 -  :math:`y_{i}` *– output of* :math:`i_{th}` neuron
    ( :math:`i_{th}` *value in output tensor).*

 -  :math:`b_{i}` *– bias for* :math:`i_{th}` *neuron*
	
Kernels which implement an RNN Dense kernel have the following prototype:

.. code::

   mli_status mli_krn_rnn_dense_<data_format>(
      const mli_tensor **inputs,
      const mli_tensor **weights,
      const mli_tensor *bias,
      const mli_rnn_dense_cfg *cfg,
      mli_tensor *out);
..	  
	  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +------------------+---------------------------+-------------------------------------------------------------------+
   | **Parameter**    | **Type**                  | **Description**                                                   |
   +==================+===========================+===================================================================+
   | ``inputs``       | ``mli_tensor **``         | [IN] Pointer to the array of pointers to constant input tensors   |
   +------------------+---------------------------+-------------------------------------------------------------------+
   | ``weights``      | ``mli_tensor **``         | [IN] Pointer to the array of pointers to constant weights tensors |
   +------------------+---------------------------+-------------------------------------------------------------------+
   | ``bias``         | ``mli_tensor *``          | [IN] Pointer to constant bias tensor                              |
   +------------------+---------------------------+-------------------------------------------------------------------+
   | ``cfg``          | ``mli_rnn_dense_cfg *``   | [IN] Pointer to RNN dense parameters structure                    |
   +------------------+---------------------------+-------------------------------------------------------------------+
   | ``out``          | ``mli_tensor *``          | [OUT] Pointer to output tensor. Result is stored here.            |
   +------------------+---------------------------+-------------------------------------------------------------------+
..

 :code:`mli_rnn_dense_cfg` is defined as:

.. code::

   typedef struct {
        uint8_t inputs_num;
    } mli_rnn_dense_cfg;
..

.. _t_mli_rnn_dense_cfg_desc:
.. table:: mli_rnn_dense_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +-----------------+--------------+------------------------------------------------------------+
   | **Field Name**  | **Type**     | **Description**                                            |
   +=================+==============+============================================================+
   |                 |              | Number of input tensors (number of pointers in inputs      |
   |                 |              | array). Also, the number of weights tensors (number of     |
   | ``inputs_num``  | ``uint8_t``  | pointers in weights   array), as each input is specified   |
   |                 |              | with its own weights tensor. Maximum number of tensors     |
   |                 |              | in the array is specified by MLI_RNN_MAX_INPUTS define.    |
   +-----------------+--------------+------------------------------------------------------------+
..

Here is a list of all available RNN Dense functions:

.. table:: List of Available RNN Dense Functions
   :align: center
   :widths: auto 
   
   +------------------------------------+--------------------------------------+
   | **Function Name**                  | **Details**                          |
   +====================================+======================================+
   | ``mli_krn_rnn_dense_sa8_sa8_sa32`` || In/out/weights data format: **sa8** |
   |                                    || Bias data format: **sa32**          |
   +------------------------------------+--------------------------------------+
   | ``mli_krn_rnn_dense_fx16``         || All tensors data format: **fx16**   |
   +------------------------------------+--------------------------------------+
   | ``mli_krn_rnn_dense_fx16_fx8_fx8`` || In/out data format: **fx16**        |
   |                                    || Weights/Bias data format: **fx8**   |
   +------------------------------------+--------------------------------------+
..

All the listed functions must comply to the following conditions :

 - ``bias``, all tensors in ``inputs`` array and all tensors in ``weights`` array 
   must be valid.
	
 - The number of tensors in inputs and ``weights`` arrays must be the same and 
   must not exceed MLI_RNN_MAX_INPUTS value. 
   
 - Each tensor in ``inputs`` array might be of any shape and rank. Only total 
   number of elements is considered. 
   
 - The i_th tensor in weights array corresponds to the i_th tensor in inputs 
   array, which means:
   
   - ``weights[i]`` must be a two-dimensional tensor of shape (N, M), where N is 
     the total number of elements in the inputs[i] tensor and M is the total 
     number of neurons and is equal to output length.   
     
 - ``bias`` must be a one-dimensional tensor. Its length must be equal to M (number 
   of filters and is equal to output length) of any weights tensor.
   
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   and valid el_params union. Other fields of the structure do not have to contain 
   valid data and are filled by the function.
   
 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions:
 
 - ``bias``, ``out``, all the tensors in inputs array, and all tensors in weights array 
   must be quantized on the tensor level. It implies that each tensor contains a 
   single scale factor and a single zero offset.
   
 - ``bias`` and all tensors in weights array must be symmetric. It implies that both 
   tensors contain single zero offset equal to 0.
   
Scale factor of bias tensor must be equal to the multiplication of scale factor of 
the **first** input and the **first** weights tensors in corresponding arrays 
(that is, bias.scale = inputs[0].scale * weights[0].scale).


Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

