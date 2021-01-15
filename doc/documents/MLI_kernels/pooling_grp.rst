Pooling group
-------------

The Pooling Group describes operations which divide input features into sub-frames 
and applies a function with scalar output on each of them. Generally, this results 
in a feature map with reduced, withheld or emphasized key features. 

Functions in this group use the mli_pool_cfg structure, defined as:

.. code::

   typedef struct {
      uint8_t kernel_width;
      uint8_t kernel_height;
      uint8_t stride_width;
      uint8_t stride_height;
      uint8_t padding_left;
      uint8_t padding_right;
      uint8_t padding_top;
      uint8_t padding_bottom;
   } mli_pool_cfg;


.. table:: mli_pool_cfg structure field description
   :align: center
   :widths: auto
   
   +----------------------+-------------+-------------------------------------------------------------------+
   | **Field name**       | **Type**    | **Description**                                                   |
   +======================+=============+===================================================================+
   | ``kernel_width``     | ``uint8_t`` | Width of the pooling kernel.                                      |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``kernel_height``    | ``uint8_t`` | Height of the pooling kernel.                                     |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``stride_width``     | ``uint8_t`` | Stride of filter across width dimension of input, is the step in  |
   |                      |             | the input tensor in the width dimension to the next filter.       |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``stride_height``    | ``uint8_t`` | Stride of filter across height dimension of input, is the step    |
   |                      |             | in the input tensor in the height dimension to the next filter.   |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``padding_left``     | ``uint8_t`` | Number of zero points implicitly added to the left of input       |
   |                      |             | (width dimension)                                                 |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``padding_right``    | ``uint8_t`` | Number of zero points implicitly added to the right of input      |
   |                      |             | (width dimension)                                                 |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``padding_top``      | ``uint8_t`` | Number of zero points implicitly added to the top of input        |
   |                      |             | (height dimension)                                                |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``padding_bottom``   | ``uint8_t`` | Number of zero points implicitly added to the bottom of input     |
   |                      |             | (height dimension)                                                |
   +----------------------+-------------+-------------------------------------------------------------------+
..


.. toctree::
   :maxdepth: 2
   
   pool_max.rst
   pool_avg.rst 