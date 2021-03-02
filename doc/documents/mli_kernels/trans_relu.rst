.. _relu_prot:

ReLU Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel represents Rectified Linear unit (ReLU). It performs various types 
of the rectifier activation on input. The following types of ReLU supported by 
this type of kernel:

 - *identity functon* :math:`y_{i} = x_{i}`
 
 - *General ReLU:* :math:`y_{i} = MAX(x_{i},\ 0)`

 - *ReLU1:* :math:`y_{i} = MAX(MIN\left( x_{i},1 \right),\  - 1)`

 - *ReLU6:* :math:`y_{i} = MAX(MIN\left( x_{i},6 \right),\ 0)`

   Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

This kernel outputs a tensor of the same shape and type as input. This kernel supports 
in-place computation: output and input can point to exactly the same memory (the same 
starting address and memory strides). 

If the starting address and memory stride of the input and output tensors are set in 
such a way that memory regions are overlapped, the behavior is undefined.

Kernels which implement ReLU functions have the following prototype:

.. code::

   mli_status mli_krn_relu_<data_format>(
      const mli_tensor *in,
      const mli_relu_cfg *cfg,
      mli_tensor *out);
..

where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the function 
parameters are shown in the following table:

.. table:: ReLU Function Parameters
   :align: center
   :widths: auto 
   
   +----------------+----------------------+----------------------------------------------------------+
   | **Parameter**  | **Type**             | **Description**                                          |
   +================+======================+==========================================================+
   | ``in``         | ``mli_tensor *``     | [IN] Pointer to constant input tensor.                   |
   +----------------+----------------------+----------------------------------------------------------+
   | ``cfg``        | ``mli_relu_cfg *``   | [IN] Pointer to relu parameters structure.               |
   +----------------+----------------------+----------------------------------------------------------+
   | ``out``        | ``mli_tensor *``     | [OUT] Pointer to output tensor. Result is stored here.   |
   +----------------+----------------------+----------------------------------------------------------+
..

   ``mli_relu_cfg`` is defined as:

.. code::
   
   typedef struct {
      mli_relu_type type;
    } mli_relu_cfg;
..

.. _t_mli_relu_cfg_desc:
.. table:: mli_relu_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +-----------------+--------------------+------------------------+-------------------------------------------------------+
   | **Field Name**  | **Type**           | **Enumeration Value**  | **Description**                                       |
   +=================+====================+========================+=======================================================+
   |                 |                    | ``MLI_RELU_NONE``      | No ReLU. Identity function                            |
   |                 |                    +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_GEN``       | General Rectifier function with output range from 0   |
   |                 | ``mli_relu_type``  |                        | to value maximum inclusively                          |
   | ``type``        | (enumeration)      +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_1``         | ReLU1 Rectifier function with output range [-1, 1]    |
   |                 |                    +------------------------+-------------------------------------------------------+
   |                 |                    | ``MLI_RELU_6``         | ReLU6 Rectifier function with output range [0, 6]     |
   +-----------------+--------------------+------------------------+-------------------------------------------------------+
..


.. table:: List of Available ReLU Functions
   :align: center
   :widths: auto 
   
   +------------------------+-----------------------------------+
   | **Function Name**      | **Details**                       |
   +========================+===================================+
   | ``mli_krn_relu_sa8``   | All tensors data format: **sa8**  |
   +------------------------+-----------------------------------+
   | ``mli_krn_relu_fx16``  | All tensors data format: **fx16** |
   +------------------------+-----------------------------------+
..

Ensure that you satisfy the following conditions before calling the function:

 - ``in`` tensor must be valid.
 
 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.
 
 - ``out`` tensor must contain a valid pointer to a buffer with sufficient capacity 
   (that is, the total amount of elements in input tensor). Other fields are filled 
   by kernel (``shape``, ``rank`` and ``el_params``).

For **sa8** versions of kernel, in addition to the preceding conditions, ensure that you 
satisfy the following condition before calling the function: 

 - ``in`` tensor must be quantized on the tensor level. It implies that the tensor 
   contains a single scale factor and a single zero offset.

Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.