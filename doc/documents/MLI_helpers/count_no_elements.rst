.. _count_no_elem:

Count Number of Elements 
~~~~~~~~~~~~~~~~~~~~~~~~~

   Function counts the number of elements in a tensor starting from the
   provided dimension number (dimension numbering starts from 0):

.. math:: num\_ of\_ elements = shape\lbrack start\_ dim\rbrack\ *shape\lbrack start\_ dim + 1\rbrack*\ldots*shape\lbrack last\_ dim\rbrack

..

   Where:

   ``num_of_elements`` - Number of accounting elements

   ``shape`` - Shape of tensor

   ``start_dim`` – Start dimension for counting

   ``last_dim`` - Last dimension of tensor (tensor rank-1)

   Function calculates total number of elements in case
   ``start_dim = 0``. Function returns 0 if conditions listed
   in the following API are violated.

.. _api-16:

API
^^^

+-----------------------+-----------------------+-------------------------------------------------+
| **Prototype**         |.. code:: c                                                              |
|                       |                                                                         |      
|                       | uint32_t mli_hlp_count_elem_num(mli_tensor *in, uint32_t start_dim)     |
+-----------------------+-----------------------+-------------------------------------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input  tensor                   |
+-----------------------+-----------------------+-------------------------------------------------+
|                       | ``start_dim``         | [IN] Start dimension for counting               |
+-----------------------+-----------------------+-------------------------------------------------+
| **Returns**           | ``Number of elements``|                                                 |
+-----------------------+-----------------------+-------------------------------------------------+

.. _conditions-for-applying-the-function-5:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Input must contain valid rank (less then ``MLI_MAX_RANK``).

-  ``start_dim`` must be less than or equal to input rank