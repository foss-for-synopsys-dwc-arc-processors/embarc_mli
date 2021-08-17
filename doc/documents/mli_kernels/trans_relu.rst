.. _relu_prot:

ReLU Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

This kernel represents Rectified Linear Unit (ReLU). It performs various types 
of the rectifier activation on input. The following types of ReLU are supported by 
this type of kernel:

 - *identity functon* :math:`y_{i} = x_{i}`
 
 - *General ReLU:* :math:`y_{i} = MAX(x_{i},\ 0)`

 - *ReLU1:* :math:`y_{i} = MAX(MIN\left( x_{i},1 \right),\  - 1)`

 - *ReLU6:* :math:`y_{i} = MAX(MIN\left( x_{i},6 \right),\ 0)`

   Where:

   :math:`x_{i}` *–* :math:`i_{\text{th}}` *value in input tensor*

   :math:`y_{i}` *–* :math:`i_{\text{th}}` *value in output tensor*

Functions
^^^^^^^^^

Kernels which implement ReLU functions have the following prototype:

.. code:: c

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
   
   +----------------+----------------------+----------------------------------------------+
   | **Parameter**  | **Type**             | **Description**                              |
   +================+======================+==============================================+
   | ``in``         | ``mli_tensor *``     | [IN] Pointer to constant input tensor.       |
   +----------------+----------------------+----------------------------------------------+
   | ``cfg``        | ``mli_relu_cfg *``   | [IN] Pointer to relu parameters structure.   |
   +----------------+----------------------+----------------------------------------------+
   | ``out``        | ``mli_tensor *``     | [IN | OUT] Pointer to output tensor.         |
   |                |                      | Result is stored here                        |
   +----------------+----------------------+----------------------------------------------+
..

   ``mli_relu_cfg`` is defined as:

.. code:: c
   
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

Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the selected version of the kernel.

 - ``in`` and ``out`` tensors must be of the same shape

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

For **sa8** versions of kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - ``in`` tensor must be quantized on the tensor level. This implies that the tensor 
   contains a single scale factor and a single zero offset.

 - Zero offset of ``in`` tensor must be within [-128, 127] range.

Ensure that you satisfy the platform-specific conditions in addition to to those listed above 
(see the :ref:`platform_spec_chptr` chapter).

Result
^^^^^^

These functions modify:

 - Memory pointed by ``out.data.mem`` field.  
 - ``el_params`` field of ``out`` tensor which is copied from ``in`` tensor.

It is assumed that all the other fields and structures are properly populated 
to be used in calculations and are not modified by the kernel.

The kernel supports in-place computation. It means that ``out`` and ``in`` tensor structures 
can point to the same memory with the same memory strides but without shift.
It can affect performance for some platforms.

.. warning::

  Only an exact overlap of starting address and memory stride of the ``in`` and ``out`` 
  tensors is acceptable. Partial overlaps result in undefined behavior.
..

Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.