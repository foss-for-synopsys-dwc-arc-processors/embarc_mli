.. _fully_con_grp:

Fully Connected Prototype and Function List 
-------------------------------------------

Description
^^^^^^^^^^^

.. _f_fully_conn_layer:
.. figure:: ../images/fully_conn_layer.png
   :align: center
   
..


This kernel implements a fully connected layer, also usually referred to as the inner 
product or dense layer.  
 
Each value of output tensor is calculated according to the following formula:

.. math:: 

   y_{i} = b_{i} + \sum_{j}^{}x_{j}*W_{i,j}
..

Where:

    :math:`x_{j}` *-* :math:`j_{\text{th}}` *value in input tensor*

    :math:`y_{i}` *- output of* :math:`i_{\text{th}}` neuron
    (:math:`i_{\text{th}}` *value in output tensor)*

    :math:`W_{i,j}` *- weight of* :math:`j_{\text{th}}\ `\ *input element
    for* :math:`i_{\text{th}}` *neuron.*

    :math:`b_{i}` *- bias for* :math:`i_{\text{th}}` *neuron*

Optionally, a saturating ReLU activation function can be applied to the result of the calculations 
during the function's execution. For more information on supported ReLU types, see :ref:`relu_prot`.  

This is a MAC-based kernel which implies accumulation. See :ref:`quant_accum_infl` for more information on related quantization aspects. 
The Number of accumulation series is equal to input size.

Functions
^^^^^^^^^

Functions that implement fully connected kernels have the following prototype:

.. code:: c

   mli_status mli_krn_fully_connected_<data_format>(
      const mli_tensor *in,
      const mli_tensor *weights,
      const mli_tensor *bias,
      const mli_fully_connected_cfg *cfg,
      mli_tensor *out);
..
  
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` 
and the function parameters are shown in the following table:

.. table:: Fully Connected Function Parameters
   :align: center
   :widths: auto 
   
   +------------------+---------------------------------+-------------------------------------------------------------+
   | **Parameter**    | **Type**                        | **Description**                                             |
   +==================+=================================+=============================================================+
   | ``in``           | ``mli_tensor *``                | [IN] Pointer to constant input tensor.                      |
   +------------------+---------------------------------+-------------------------------------------------------------+
   | ``weights``      | ``mli_tensor *``                | [IN] Pointer to constant weights tensor.                    |
   +------------------+---------------------------------+-------------------------------------------------------------+
   | ``bias``         | ``mli_tensor *``                | [IN] Pointer to constant bias tensor.                       |
   +------------------+---------------------------------+-------------------------------------------------------------+
   | ``cfg``          | ``mli_fully_connected_cfg *``   | [IN] Pointer to fully connected parameters structure.       |
   +------------------+---------------------------------+-------------------------------------------------------------+
   | ``out``          | ``mli_tensor *``                | [IN | OUT] Pointer to output tensor. Result is stored here. |
   +------------------+---------------------------------+-------------------------------------------------------------+
..

   ``mli_fully_connected_cfg`` is defined as:

.. code:: c
   
   typedef struct {
        mli_relu_cfg relu;
   } mli_fully_connected cfg; 
..

.. _t_mli_fc_cfg_desc:
.. table:: mli_fully_connected_cfg Structure field description
   :align: center
   :widths: auto 
   
   +-----------------+--------------------+-------------------------------------------------------+
   | **Field Name**  | **Type**           | **Description**                                       |
   +=================+====================+=======================================================+
   |                 |                    | Type of ReLU activation applied to output values.     |
   | ``relu``        | ``mli_relu_cfg``   | See :ref:`relu_prot` for definition of this structure |
   +-----------------+--------------------+-------------------------------------------------------+
..

Here is a list of all available Fully Connected functions:

.. table:: List of Available Fully Connected Functions
   :align: center
   :widths: auto 
   
   +---------------------------------------------------+----------------------------------------+
   | **Function Name**                                 | **Details**                            |
   +===================================================+========================================+
   | ``mli_krn_fully_connected_sa8_sa8_sa32``          || In/out/weights data format: **sa8**   |
   |                                                   || Bias data format: **sa32**            |
   +---------------------------------------------------+----------------------------------------+
   | ``mli_krn_fully_connected_fx16``                  || All tensors data format: **fx16**     |
   +---------------------------------------------------+----------------------------------------+
   | ``mli_krn_fully_connected_fx16_fx8_fx8``          || In/out data format: **fx16**          |
   |                                                   || Weights/Bias data format: **fx8**     |
   +---------------------------------------------------+----------------------------------------+
   | ``mli_krn_fully_connected_sa8_sa8_sa32_ext_bias`` || In/out/weights data format: **sa8**   |
   |                                                   || Bias data format: **sa32**            |
   |                                                   || Bias data adjusted to include         |
   |                                                   || zero point additives                  |
   +---------------------------------------------------+----------------------------------------+
..

``mli_krn_fully_connected_sa8_sa8_sa32_ext_bias`` is a specialized version of 
``mli_krn_fully_connected_sa8_sa8_sa32`` which performs calculations much faster, but requires bias
data to be adjusted according to the following formula:

.. math:: 

   \hat{b}_{i} = b_{i} + \sum_{j}^{}in\_zp*W_{i,j}
..

Where:

    :math:`in\_zp` *–* zero point of input sa8 tensor

    :math:`W_{i,j}` *– weight of* :math:`j_{\text{th}}\ `\ *input element
    for* :math:`i_{\text{th}}` *neuron.*

    :math:`b_{i}` *– original sa32 bias for* :math:`i_{\text{th}}` *neuron*
 
    :math:`\hat{b}_{i}` *– adjusted sa32 bias for* :math:`i_{\text{th}}` *neuron*

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in``, ``out``, ``weights`` and ``bias`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the selected version of the kernel.

 - Shapes of ``in``, ``out``, ``weights`` and ``bias`` tensors must be compatible,
   which implies the following requirements:

    - ``in`` tensor might be of any shape and rank. Only total number of elements is 
      considered.

    - ``weights`` is a 2-dimensional tensor (rank==2) of shape :math:`(N, M)`, where 
      :math:`N` is the total number of elements in the input tensor and :math:`M`
      is the total number of neurons and is equal to output length.

    - ``bias`` must be a one-dimensional tensor (rank==1). Its length must be equal to 
      :math:`M` dimension (number of filters and is equal to output length) of weights tensor.

    - ``out`` must be a one-dimensional tensor (rank==1). Its length must be equal to 
      :math:`M` dimension (number of filters) of weights tensor.

 - ``in`` and ``out`` tensors must not point to overlapped memory regions.
   
 - ``mem_stride`` must satisfy the following statements:
   
    - For ``in`` and ``out`` tensors - memstride must reflect the shape, 
      e.g memory of these tensors must be contiguous.
      
    - For ``weights`` and ``bias`` tensor - memstride of the innermost dimension must 
      be equal to 1.

For **fx16** and **fx16_fx8_fx8** versions of kernel, in addition to the general conditions, ensure that you 
satisfy the following quantization conditions before calling the function:

 - The number of ``frac_bits`` in the ``bias`` and ``out`` tensors must not exceed the sum of ``frac_bits`` 
   in the ``in`` and ``weights`` tensors.
 
For **sa8_sa8_sa32** versions of kernel, in addition to the general conditions, ensure that you 
satisfy the following quantization conditions before calling the function: 

 - ``in`` and  ``out`` tensors must be quantized on the tensor level. 
   It implies that each tensor contains a single scale factor and a single zero offset.
   
 - Zero offset of ``in`` and ``out`` tensors must be within [-128, 127] range.

 - ``weights`` and ``bias`` tensors must be symmetric. Both must be quantized at the same level.
   Allowed options are
   
    - Per Tensor level. This implies that each tensor contains a single scale factor and a single zero
      offset equal to 0.
      
    - Per :math:`M` dimension level (number of neurons). This implies that each tensor contains separate scale point
      for each sub-tensor. All tensors contain single zero offset equal to 0.
   
 - Scale factors of bias tensor must be equal to the multiplication of input scale factor 
   broadcasted on weights array of scale factors. See the example for the similar condition 
   in the :ref:`conv_2d`.

Ensure that you satisfy the platform-specific conditions in addition to to those listed above 
(see the :ref:`platform_spec_chptr` chapter).

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the other fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

