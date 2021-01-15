.. _fully_con_grp:

Fully Connected Prototype and Function List 
-------------------------------------------

This kernel implements fully connected layer, also usually referred to as the inner 
product or dense layer, as shown in Figure :ref:`f_fully_conn_layer`.  
 
.. _f_fully_conn_layer:
.. figure:: ../images/fully_conn_layer.png
   :align: center
   
   Fully Connected Layer
..


Each value of output tensor is calculated according to the following formula:

.. math:: 

   y_{i} = b_{i} + \sum_{j}^{}x_{j}*W_{i,j}
..

Where:

 -  :math:`x_{j}` *–* :math:`j_{\text{th}}` *value in input tensor*

 -  :math:`y_{i}` *– output of* :math:`i_{\text{th}}` neuron
    (:math:`i_{\text{th}}` *value in output tensor)*

 -  :math:`W_{i,j}` *– weight of* :math:`j_{\text{th}}\ `\ *input element
    for* :math:`i_{\text{th}}` *neuron.*

 -  :math:`b_{i}` *– bias for* :math:`i_{\text{th}}` *neuron*
	
Functions which implement fully connected kernels have the following prototype:

.. code::

   mli_status mli_krn_fully_connected_<data_format>(
      const mli_tensor *in,
      const mli_tensor *weights,
      const mli_tensor *bias,
      mli_tensor *out);
..
  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` 
and the function parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +------------------+--------------------+--------------------------------------------------------+
   | **Parameter**    | **Type**           | **Description**                                        |
   +==================+====================+========================================================+
   | ``in``           | ``mli_tensor *``   | [IN] Pointer to constant input tensor.                 |
   +------------------+--------------------+--------------------------------------------------------+
   | ``weights``      | ``mli_tensor *``   | [IN] Pointer to constant weights tensor.               |
   +------------------+--------------------+--------------------------------------------------------+
   | ``bias``         | ``mli_tensor *``   | [IN] Pointer to constant bias tensor.                  |
   +------------------+--------------------+--------------------------------------------------------+
   | ``out``          | ``mli_tensor *``   | [OUT] Pointer to output tensor. Result is stored here. |
   +------------------+--------------------+--------------------------------------------------------+
..

Here is a list of all available Fully Connected functions:

.. table:: List of Available Fully Connected Functions
   :align: center
   :widths: auto 
   
   +------------------------------------------+----------------------------------------+
   | **Function Name**                        | **Details**                            |
   +==========================================+========================================+
   | ``mli_krn_fully_connected_sa8_sa8_sa32`` || In/out/weights data format: **sa8**   |
   |                                          || Bias data format: **sa32**            |
   +------------------------------------------+----------------------------------------+
   | ``mli_krn_fully_connected_fx16``         || All tensors data format: **fx16**     |
   +------------------------------------------+----------------------------------------+
   | ``mli_krn_fully_connected_fx16_fx8_fx8`` || In/out data format: **fx16**          |
   |                                          || Weights/Bias data format: **fx8**     |
   +------------------------------------------+----------------------------------------+
..

All the listed functions must comply to the following conditions:

 - ``in``, ``weights`` and ``bias`` tensors must be valid.
 
 - ``in`` tensor might be of any shape and rank. Only total number of elements is 
   considered.
   
 - ``weights`` must be a two-dimensional tensor of shape (N, M), where N is the 
   total number of elements in the input tensor and M is the total number of 
   neurons and is equal to output length
   
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   and valid ``el_params`` union. Other fields of the structure do not have to contain 
   valid data and are filled by the function.
   
 - ``bias`` must be a one-dimensional tensor. Its length must be equal to M dimension 
   (number of filters and is equal to output length) of weights tensor.
   
 - ``in`` and ``out`` tensors must not point to overlapped memory regions.

 - ``mem_stride`` of the innermost dimension should be equal to 1 for all the tensors.
 
For **sa8_sa8_sa32** versions of kernel, in addition to the preceding conditions: 

 - ``in``, ``out``, ``weights`` and ``bias`` tensors must be quantized on the tensor level. 
   It implies that each tensor contains a single scale factor and a single zero offset.
   
 - ``weights`` and ``bias`` tensors must be symmetric. It implies that both tensors contain 
   single zero offset equal to 0.
   
 - Scale factor of bias tensor must be equal to the multiplication of input scale factor 
   and weights scale factor.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

