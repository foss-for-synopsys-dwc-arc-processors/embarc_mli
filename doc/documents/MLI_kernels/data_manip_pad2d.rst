.. _pad_2d:

Padding 2D
~~~~~~~~~~

This kernel performs zero padding of borders across height and width
dimensions of vision-specific input feature maps (see :ref:`fns`).

Padding for each side of image (top, bottom, left, right) is
configured separately according to input configuration structure, but
the same padding for each side is used across all channels. Padding
for HWC and CHW layouts of input tensor is implemented as separate
functions. Output is calculated in the following order:

.. math:: out\_ channels = in\_ channels 

.. math:: out\_ height = in\_ height\  + padding\_ top + padding\_ bottom

.. math:: out\_ width = in\_ width\  + padding\_ left + padding\_ right

..

For example, input tensor of the shape [2, 4, 8] representing image
of CHW layout (channel dimension is the first value, so the image
consists of 2 channels, 4 rows and 8 columns each). After applying
the padding on the top and the right, with the ``padding_top=2`` and
``padding right=1``, an output image of shape [2, 6, 9] is produced,
where top 2 rows contains only zeros, and last value of each row
also equal to zero.

.. note::
   Ensure that input and output
   tensors do not point to     
   overlapped memory regions,  
   otherwise the behavior is   
   undefined.                

.. _function-configuration-structure-18:

Function Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Definition
''''''''''

.. code:: c                
                           
   typedef struct {          
      uint8_t padding_left;  
      uint8_t padding_right; 
      uint8_t padding_top;   
      uint8_t padding_bottom;
   }  mli_padding2d_cfg;     
..
   
Parameters
''''''''''

.. table:: Function Configuration Parameters
   :widths: 20,130
   
   +-----------------------+-----------------------+
   | **Fields**            | **Description**       |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``padding_left``      | Number of zero points |
   |                       | added to the left     |
   |                       | side of input (width  |
   |                       | dimension).           |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``padding_right``     | Number of zero points |
   |                       | added to the right    |
   |                       | side of input (width  |
   |                       | dimension).           |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``padding_top``       | Number of zero points |
   |                       | added to the upper    |
   |                       | side of input (height |
   |                       | dimension).           |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``padding_bottom``    | Number of zero points |
   |                       | added to the bottom   |
   |                       | side of input (height |
   |                       | dimension).           |
   +-----------------------+-----------------------+

.. _api-14:

Kernel Interface
^^^^^^^^^^^^^^^^

Prototype
'''''''''

.. code:: c                      
                                 
 mli_status mli_krn_permute_fx ( 
    const mli_tensor *in,        
    const mli_padding2d_cfg *cfg,
    mli_tensor *out);            
..

Parameters
''''''''''
	
.. table:: Kernel Interface Parameters
   :widths: 20,130
   
   +-----------------------+-----------------------+
   | **Parameters**        | **Description**       |
   +=======================+=======================+
   |                       |                       |
   | ``in``                | [IN] Pointer to input |
   |                       | tensor                |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``cfg``               | [IN] Pointer to       |
   |                       | configuration         |
   |                       | structure with        |
   |                       | padding parameters    |
   +-----------------------+-----------------------+
   |                       |                       |
   | ``out``               | [OUT] Pointer to the  |
   |                       | output tensor. Result |
   |                       | is stored here        |
   +-----------------------+-----------------------+

.. _kernel-specializations-14:

Kernel Specializations
^^^^^^^^^^^^^^^^^^^^^^

.. table:: Non-Specialized Functions
   :widths: 20,130
   
   +--------------------------------+--------------------------------------+
   | **Function**                   | **Description**                      |
   +================================+======================================+
   ||                      *HWC Data Layout*                               |
   +--------------------------------+--------------------------------------+
   | ``mli_krn_padding2d_hwc_fx8``  | General function; 8bit FX elements;  |
   +--------------------------------+--------------------------------------+
   | ``mli_krn_padding2d_hwc_fx16`` | General function; 16bit FX elements; |
   +--------------------------------+--------------------------------------+
   ||                      *СHW Data Layout*                               |
   +--------------------------------+--------------------------------------+
   | ``mli_krn_padding2d_сhw_fx8``  | General function; 8bit FX elements;  |
   +--------------------------------+--------------------------------------+
   | ``mli_krn_padding2d_сhw_fx16`` | General function; 16bit FX elements; |
   +--------------------------------+--------------------------------------+

.. _conditions-for-applying-the-kernel-14:

Conditions for Applying the Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure that you satisfy the following conditions before applying the
function:

-  Input, tensors must be valid (see :ref:`mli_tns_struct`)

-  Before processing, output tensor must contain a valid pointer to a
   buffer with sufficient capacity enough for storing the result
   (that is, the total amount of elements in input tensor). Other
   fields are filled by kernel (shape, rank and element specific
   parameters)

-  Buffers of input and output tensors must point to different
   not-overlapped memory regions