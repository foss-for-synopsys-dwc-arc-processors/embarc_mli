.. _get_elm_size:

Get Basic Element Size
~~~~~~~~~~~~~~~~~~~~~~

This function returns size of tensor basic element in bytes. It
returns 0 if conditions listed the following API are violated.

.. _api-15:

Kernel Interface
^^^^^^^^^^^^^^^^

Prototype
'''''''''

.. code:: c                      
                                 
   uint32_t mli_hlp_count_elem_num (mli_tensor *in)                
..

Parameters
''''''''''

.. table:: Kernel Interface Parameters
   :widths: 20,130
   
   +-----------------------+-----------------------+
   | **Parameters**        | **Description**       |   
   +=======================+=======================+
   | ``in``                | [IN] Pointer to input |
   |                       | tensor                |
   +-----------------------+-----------------------+

**Returns**  - Size of basic element in bytes     
	 
.. _conditions-for-applying-the-function-4:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function must point to the tensor of supported element type (see
:ref:`mli_elm_enum`).

