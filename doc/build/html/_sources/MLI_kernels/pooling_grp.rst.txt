.. _chap_pool:

Pooling Kernels Group
=====================

The Pooling Group describes operations which divide input features into sub-frames 
and applies a function with scalar output on each of them. Generally, this results 
in a feature map with reduced, withheld, or emphasized key features. 

Functions in this group use the ``mli_pool_cfg`` structure, defined as:

.. code:: c

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

.. tabularcolumns:: |\Y{0.3}|\Y{0.15}|\Y{0.45}|

.. _t_mli_pool_cfg_desc:
.. table:: mli_pool_cfg structure field description
   :align: center
   :class: longtable
   
   +----------------------+-------------+-------------------------------------------------------------------+
   | **Field name**       | **Type**    | **Description**                                                   |
   +======================+=============+===================================================================+
   | ``kernel_width``     | ``uint8_t`` | Width of the pooling kernel.                                      |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``kernel_height``    | ``uint8_t`` | Height of the pooling kernel.                                     |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``stride_width``     | ``uint8_t`` | Stride of filter across width dimension of input; is the step in  |
   |                      |             | the input tensor in the width dimension to the next filter.       |
   +----------------------+-------------+-------------------------------------------------------------------+
   | ``stride_height``    | ``uint8_t`` | Stride of filter across height dimension of input; is the step    |
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

For all kernels in this group, spatial dimensions of
``in`` and ``out`` tensors (Width and Height) must comply with the following 
system of equations:

.. math::
   :label: eq_pool_shapes

   \begin{cases}

   \hat{Wi} = {Wi}+padding\_left+padding\_right

   \hat{Hi} = {Hi}+padding\_top+padding\_bottom

   {Wo}*{stride\_width} = \hat{Wi}-{kernel\_width}+1

   {Ho}*{stride\_height} = \hat{Hi}-{kernel\_height}+1

   \end{cases}
..

Where:

   - :math:`\hat{Wi}`, :math:`\hat{Hi}` *- effective* ``in`` *feature map width and height
     after applying* :math:`padding\_*` *to the original width* (:math:`Wi`) *and height* (:math:`Hi`).

   - :math:`Wo`, :math:`Ho` *-* ``out`` *feature map width and height.*

.. toctree::
   :maxdepth: 1
   
   pool_max.rst
   pool_avg.rst
