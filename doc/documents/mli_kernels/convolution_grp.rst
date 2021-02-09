Convolution Group
-----------------

The Convolution Group describes operations related to convolution layers, where 
input features are convolved with a set of trained filters. For a comprehensive guide 
on convolution arithmetic details for various cases, see [2]. 

Functions in this group use the ``mli_conv2d_cfg`` structure, defined as:

.. code::

  typedef struct {
     mli_relu_cfg relu;
     uint8_t stride_width;
     uint8_t stride_height;
     uint8_t dilation_width;
     uint8_t dilation_height;
     uint8_t padding_left;
     uint8_t padding_right;
     uint8_t padding_top;
     uint8_t padding_bottom; 
  } mli_conv2d_cfg;
..

.. _t_mli_conv2d_cfg_desc:
.. table:: mli_conv2d_cfg Structure Field Description
   :align: center
   :widths: 30, 50, 130 
   
   +-----------------------+---------------------+---------------------------------------------------+
   | **Field Name**        | **Type**            | *Description**                                    |
   +=======================+=====================+===================================================+
   | ``relu``              | ``mli_relu_cfg``    | Type of ReLU activation applied to output values. | 
   |                       |                     | See :ref:`relu_prot` for more details.            |  
   +-----------------------+---------------------+---------------------------------------------------+
   | ``stride_width``      | ``uint8_t``         | Stride of filter across width dimension of input  |
   +-----------------------+---------------------+---------------------------------------------------+
   | ``stride_ height``    | ``uint8_t``         | Stride of filter across height dimension of input |
   +-----------------------+---------------------+---------------------------------------------------+
   | ``dilation_width*``   | ``uint8_t``         | If set to k>1, there are k-1 implicitly added     |
   |                       |                     | zero points between each filter point across      |
   |                       |                     | width dimension. If set to 0 or 1, no dilation    |
   |                       |                     | logic is used.                                    |
   +-----------------------+---------------------+---------------------------------------------------+
   | ``dilation_height*``  | ``uint8_t``         | If set to k>1, there are k-1 implicitly added     |
   |                       |                     | zero points between each filter point across      |
   |                       |                     | height dimension. If set to 0 or 1, no dilation   |
   |                       |                     | logic is used.                                    |
   +-----------------------+---------------------+---------------------------------------------------+   
   | ``padding_left``      | ``uint8_t``         | Number of zero points implicitly added to the     |
   |                       |                     | left of input (width dimension)                   |
   +-----------------------+---------------------+---------------------------------------------------+   
   | ``padding_right``     | ``uint8_t``         | Number of zero points implicitly added to the     |
   |                       |                     | right of input (width dimension)                  |
   +-----------------------+---------------------+---------------------------------------------------+   
   | ``padding_top``       | ``uint8_t``         | Number of zero points implicitly added to the     |
   |                       |                     | top of input (height dimension)                   |
   +-----------------------+---------------------+---------------------------------------------------+
   | ``padding_bottom``    | ``uint8_t``         | Number of zero points implicitly added to the     |
   |                       |                     | bottom of input (height dimension)                |
   +-----------------------+---------------------+---------------------------------------------------+ 
..

For more info on dilation rate, see chapter 5.1 of [2].

.. toctree::
   :maxdepth: 2
   
   conv_2d.rst
   conv_depthwise.rst 
   conv_transp.rst
   conv_grp.rst
   
   



   
   

