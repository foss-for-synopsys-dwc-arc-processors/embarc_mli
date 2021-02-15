.. _err_codes:

Error Codes
-----------

Most functions return a value of type mli_status. This is an enumeration type with fields 
as described in table :ref:`t_mli_status_enum`.

.. _t_mli_status_enum:
.. table:: mli_status Enum Fields
   :align: center
   :widths: 50, 50, 130 
   
   +-------------------------------------+----------------+--------------------------------------------------------------------------+
   | **Field name**                      | **value**      | **Description**                                                          |
   +=====================================+================+==========================================================================+
   | ``MLI_STATUS_OK``                   | ``0``          | No error occurred                                                        |      
   +-------------------------------------+----------------+--------------------------------------------------------------------------+
   | ``MLI_STATUS_BAD_TENSOR``           |                | Invalid tensor is passed to the function                                 |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+
   | ``MLI_STATUS_SHAPE_MISMATCH``       |                | The shapes or rank of the tensors are not compatible for this function   |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_INCOMPATEBLE_TENSORS`` |                | Some parameters of the tensors are not compatible for this function      |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_BAD_FUNC_CFG``         |                | Invalid configuration is passed to the function                          |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_NOT_ENGH_MEM``         |                | Capacity of the output tensor is not enough for the function result      |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_NOT_SUPPORTED``        |                | Function is not implemented for this combination of inputs               |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_SPEC_PARAM_MISMATCH``  |                | Configuration of the function does not match the specialization          |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+ 
   | ``MLI_STATUS_ARGUMENT_ERROR``       |                | A passed by reference parameter is NULL                                  |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+    
   | ``MLI_STATUS_TYPE_MISMATCH``        |                | The datatype of the tensors does not match the function specialization   | 
   +-------------------------------------+----------------+--------------------------------------------------------------------------+   
   | ``MLI_STATUS_LARGE_ENUM``           | ``0x02000000`` | Dummy field to prevent size optimizations of public enums                |
   +-------------------------------------+----------------+--------------------------------------------------------------------------+
..
  