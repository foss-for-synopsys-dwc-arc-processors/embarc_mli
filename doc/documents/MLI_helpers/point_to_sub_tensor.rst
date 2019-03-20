.. _point_sub_tensor:

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