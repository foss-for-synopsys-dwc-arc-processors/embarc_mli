Basic Functions 
----------------

   This is a set of utility functions for getting information from data
   structures and performing various operations on it.

Get Basic Element Size
~~~~~~~~~~~~~~~~~~~~~~

   This function returns size of tensor basic element in bytes. It
   returns 0 if conditions listed the following API are violated.

.. _api-15:

API
^^^

+-----------------------+-----------------------+-----------------------+
|                       |.. code:: c                                    |
|                       |                                               |
| **Prototype**         | uint32_t mli_hlp_count_elem_num               |
|                       | (mli_tensor *in)                              |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
|                       |                       |                       |
| **Parameters**        | ``in``                | [IN] Pointer to input |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+
| **Returns**           | Size of basic element                         |
|                       | in bytes                                      |
+-----------------------+-----------------------+-----------------------+

.. _conditions-for-applying-the-function-4:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   The function must point to the tensor of supported element type (see
   :ref:`mli_elm_enum`).

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

Point to Sub-Tensor
~~~~~~~~~~~~~~~~~~~

   This function points to sub tensors in input tensor. This can be
   considered as indexing in a multidimensional array without copying.

   For example, given a feature map IN in CHW layout of shape [8, 4,
   16], the function returns several sequential channels as output
   tensor without copying the data. Comparing with python syntax, this
   is similar to slice IN[2:4][:][:] and results in a tensor of shape
   [2,4,16] that points to channels #2 and #3.

   In terms of configuration for considering functions (see 
   :ref:`fn_conf_psubt`):

   ``start_coord = {2}; coord_num=1; first_out_dim_size=2;``

   In the same way, rows can be obtained from one of the channels. IN[3][2:3][:] 
   results in tensor of shape [1,16] (row #2 of channel#3). 
   In terms of configuration for considering functions:

   ``start_coord = {3, 2}; coord_num=2; first_out_dim_size=1;``

   This function performs operations on pointers and does not copy data
   (only points to subsequent data in input). For this reason, this
   function takes only parameters that can be translated to starting
   coordinates and size of required data.

.. _fn_conf_psubt:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------+-----------------------------------------------+
| **Definition**        |.. code:: c                                    |
|                       |                                               |                         
|                       | typedef struct {                              |                                           
|                       | uint32_t                                      |
|                       | start_coord[MLI_MAX_RANK ];                   |
|                       | uint8_t coord_num;                            |
|                       | uint8_t first_out_dim_size;                   |
|                       | }                                             |
|                       | mli_point_to_subtsr_cfg;                      |
|                       |                                               |
+-----------------------+-----------------------+-----------------------+
| **Fields**            | ``start_coord``       | Coordinates to sub    |
|                       |                       | tensor. Each          |
|                       |                       | coordinate            |
|                       |                       | corresponds to the    |
|                       |                       | dimension of the      |
|                       |                       | input tensor starting |
|                       |                       | from the first one.   |
+-----------------------+-----------------------+-----------------------+
|                       | ``coord_num``         | Number of coordinates |
+-----------------------+-----------------------+-----------------------+
|                       | ``first_out_dim_size``| Size of the first     |
|                       |                       | dimension in output   |
|                       |                       | tensor                |
+-----------------------+-----------------------+-----------------------+

.. _api-17:

API
^^^

+-----------------------+----------------------------------------------------------------------------------+
| **Prototype**         | .. code:: c                                                                      |
|                       |                                                                                  | 
|                       |   mli_status mli_hlp_point_to_subtensor(                                         |
|                       |     const mli_tensor *in, const mli_point_to_subtsr_cfg *cfg, mli_tensor *out);  |
+-----------------------+-----------------------+----------------------------------------------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input tensor                             |
+-----------------------+-----------------------+----------------------------------------------------------+
|                       | ``cfg``               | [IN] Pointer to the function configuration structure     |
+-----------------------+-----------------------+----------------------------------------------------------+
|                       | ``out``               | [OUT] Pointer to output tensor for storing the result    |
+-----------------------+-----------------------+----------------------------------------------------------+

.. _conditions-for-applying-the-function-6:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   -  Input must be a valid tensor (see :ref:`mli_tns_struct`)

   -  out must point to tensor structure.

   -  Configuration structure fields have the following conditions:

      -  The number of coordinates must be less than the input tensor rank

      -  Each coordinate must be less than the corresponding input
         dimension size

   -  Sum of ``first_dim_size`` field and the last coordinate must be less
         than or equal to the corresponding dimension in input tensor

Convert Tensor
~~~~~~~~~~~~~~

   This function copies elements from input tensor to output with data
   conversion according to the output tensor type parameters.

   For example, the function can:

   -  convert data according to new element type: ``fx16`` to ``fx8`` and backward

   -  change data according to new data parameter: increase/decrease the
      number of fractional bits while keeping the same element type for
      FX data

..

   Conversion is performed using

   -  rounding when the number of significant bits increases.

   -  saturation when the number of significant bits decreases.

..

   This operation does not change tensor shape. It copies it from input
   to output.

   Kernel can perform in-place computation, but only for conversions
   without increasing data size, so that that it does not lead to
   undefined behavior. Therefore, output and input might point to exactly the
   same memory (but without shift) except ``fx8`` to ``fx16`` conversion.
   In-place computation might affect performance for some platforms.

.. _api-18:

API
^^^

+-----------------------+-----------------------+----------------------------------------------+
| **Prototype**         |.. code:: c                                                           |
|                       |                                                                      |
|                       | mli_status mli_hlp_convert_tensor(mli_tensor *in, mli_tensor *out);  |
|                       |                                                                      |
+-----------------------+-----------------------+----------------------------------------------+
| **Parameters**        | ``in``                | [IN] Pointer to input                        |
|                       |                       | tensor                                       |
+-----------------------+-----------------------+----------------------------------------------+
|                       | ``start_dim``         | [OUT] Pointer to                             |
|                       |                       | output tensor                                |
+-----------------------+-----------------------+----------------------------------------------+
| **Returns**           | ``status code``       |                                              |
+-----------------------+-----------------------+----------------------------------------------+

.. _conditions-for-applying-the-function-7:

Conditions for Applying the Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   -  Input must be a valid tensor (see :ref:`mli_tns_struct`).

   -  Before processing the output tensor must contain a valid pointer to a
      buffer with sufficient capacity enough for storing the result
      (that is, the total amount of elements in input tensor).

   -  The output tensor also must contain valid element type and its
      parameter (``el_params.fx.frac_bits``)

   -  Before processing, the output tensor does not have to contain valid
      shape and rank - they are copied from input tensor.

