.. _err_codes:

Error Codes
-----------

Functions return value of *mli_status* enumeration type which is
declared in **include/mli_types.h**. By default, functions do not
validate inputs and typically return only ``MLI_STATUS_OK``.

To turn on the checking logic, ensure that you build the MLI library
with along with the required debug mode as described in section
:ref:`func_param_dbg`. This might slightly affect the performance and code size of the library.

:ref:`mli_status_val_desc` contains list of status code with description.

.. _mli_status_val_desc:
.. table:: mli_status Values Description
   :widths: auto   

   +-----------------------------------+-----------------------------------+
   | Value                             | Field Description                 |
   +===================================+===================================+
   | ``MLI_STATUS_OK``                 | No error occurred                 |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_BAD_TENSOR``         | Invalid tensor is passed to the   |
   |                                   | function                          |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_SHAPE_MISMATCH``     | Shape of tensors are not          |
   |                                   | compatible for the function       |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_BAD_FUNC_CFG``       | Invalid configuration structure   |
   |                                   | is passed                         |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_NOT_ENGH_MEM``       | Capacity of output tensor is not  |
   |                                   | enough for function result        |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_NOT_SUPPORTED``      | Function is not yet implemented,  |
   |                                   | or inputs combination is not      |
   |                                   | supported                         |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_SPEC_PARAM_MISMATCH``| Function parameters do not match  |
   |                                   | the one specified in the          |
   |                                   | specialized function              |
   +-----------------------------------+-----------------------------------+

