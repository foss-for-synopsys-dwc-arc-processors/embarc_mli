.. _data:

Data
----

.. _data_types:

Data Types
~~~~~~~~~~


This library is intended to work with structured data types. Each data type groups related parameters in one entity as defined in .\ **/include/mli_types.h** .



.. _mli_tns_struct:
   
mli_tensor Structure
^^^^^^^^^^^^^^^^^^^^^

``mli_tensor`` is the main container type for input and output data that
must be processed by ML algorithm. ``mli_tensor`` represents
multi-dimensional arrays of a particular shape and includes not only
a data, but also its shape, type, and other data-specific parameters.

.. note::
   The term “data” includes input
   features, output features,    
   layer weights, and biases. It 
   does not include layer        
   parameters such as padding or 
   stride for convolutional      
   primitives. This data         
   representation is common for  
   neural networks and other     
   machine learning tasks.       
	  
.. code:: c

   typedef struct _mli_tensor {
      void *data;
      uint32_t capacity;
      uint32_t shape[MLI_MAX_RANK];
      uint32_t rank;
      mli_element_type el_type;
      mli_element_params el_params;
   } mli_tensor;

   
.. _mli_tensor_struc:
.. table:: mli_tensor Structure Fields
   :widths: 15,15,70
   
   +-----------------------+-----------------------+-----------------------+
   | Field Name            | Field Type            | Field Description     |
   +=======================+=======================+=======================+
   |    ``data``           |    ``void *``         | Pointer to memory     |
   |                       |                       | with tensor data      |
   +-----------------------+-----------------------+-----------------------+
   |    ``capacity``       |    ``uint32_t``       | Size in bytes of the  |
   |                       |                       | available memory,     |
   |                       |                       | allocated at the      |
   |                       |                       | address in data field |
   +-----------------------+-----------------------+-----------------------+
   |    ``shape``          |    ``uint32_t[]``     | Array with tensor     |
   |                       |                       | dimensions.           |
   |                       |                       | Dimensions must be    |
   |                       |                       | stored in direct      |
   |                       |                       | order starting from   |
   |                       |                       | the one with the      |
   |                       |                       | highest stride        |
   |                       |                       | between data          |
   |                       |                       | portions. For         |
   |                       |                       | example, for a matrix |
   |                       |                       | of shape              |
   |                       |                       | [rows][columns],      |
   |                       |                       | shape[0] = rows and   |
   |                       |                       | shape[1] = columns    |
   +-----------------------+-----------------------+-----------------------+
   |    ``rank``           |    ``uint32_t``       | Tensor rank with      |
   |                       |                       | countable number of   |
   |                       |                       | dimensions. Must be   |
   |                       |                       | less or equal to      |
   |                       |                       | value of MLI_MAX_RANK |
   +-----------------------+-----------------------+-----------------------+
   |    ``el_type``        |    ``mli_element``    | Type of elements      |
   |                       |                       | stored in the tensor. |
   |                       |                       | For details, see      |
   |                       |                       | :ref:`mli_elm_enum`   |
   +-----------------------+-----------------------+-----------------------+
   |    ``el_params``      | ``mli_element_params``| Parameters of         |
   |                       |                       | elements stored in    |
   |                       |                       | tensor. For details,  |
   |                       |                       | see                   |
   |                       |                       | :ref:`mli_el_prm_u`   |
   +-----------------------+-----------------------+-----------------------+

..

Some primitives, such as slope coefficient for leaky ReLU, might
require scalar parameters which explicitly involve calculations (not
primitive configuration parameter). Such scalar
inputs/outputs are also passed to primitive as tensor of rank=1 and
shape[0]=1. For some primitives it is possible to provide scalar
value as tensor with zero rank and single value stored directly in
data field (not pointed by it).

All input tensors passed to primitives typically must be valid with
fields that are not conflicting. Examples of field conflicts are:

-  Tensor shape declares more data than tensors capacity

-  Tensor rank is bigger than MLI_MAX_RANK

-  Data pointer is invalid (NULL pointer) except cases when it is
   allowed to pass scalar value directly in data field

-  Tensor is empty (total number of elements is zero)

.. _mli_elm_enum:
   
mli_element_type Enumeration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``mli_element_type`` defines basic element type stored in tensor
structure (data field).

.. code:: c
   
   typedef enum {
      MLI_EL_FX_8 = 0,
      MLI_EL_FX_16,
   } mli_element_type;

\

.. _mli_element_type_val_desc:
.. table:: mli_element_type Values Description
   :widths: auto
   
   +-----------------------------------+-----------------------------------+
   |    Value                          |    Field Description              |
   +===================================+===================================+
   | ``MLI_EL_FX_8``                   | 8-bit deep, fixed-point data with |
   |                                   | configurable number of fractional |
   |                                   | bits (see :ref:`mli_fpd_fmt`).    |
   |                                   | Data container is int8_t.         |
   |                                   | Also mentioned in this            |
   |                                   | document as fx8 type.             |
   +-----------------------------------+-----------------------------------+
   | ``MLI_EL_FX_16``                  | 16-bit deep fixed-point data with |
   |                                   | configurable number of fractional |
   |                                   | bits (see :ref:`mli_fpd_fmt`).    |
   |                                   | Data container is int16_t.        |
   |                                   | Also mentioned in this            |
   |                                   | document as fx16 type.            |
   +-----------------------------------+-----------------------------------+

.. _mli_el_prm_u:   
   
mli_element_params Union
^^^^^^^^^^^^^^^^^^^^^^^^

``mli_element_params`` union stores data type parameters required for
arithmetical operations with tensor elements.

.. code:: c
   
   typedef union _mli_element_params {
      struct{
         unsigned frac_bits;
      } fx;
   } mli_element_params;

..

Parameters are wrapped into union for future library extensibility.
The current version supports only fixed point data with configurable
number of fractional bits (see :ref:`mli_fpd_fmt`) and union
can be interpreted only as the following structure.
   
\

.. _mli_element_params_struct_fields:
.. table:: mli_element_params Structure Fields
   :widths: auto   

   +-----------------+-----------------+-----------------+-----------------+
   | Union           | Field           | Field Type      | Field           |
   | Interpretation  |                 |                 | Description     |
   +=================+=================+=================+=================+
   | ``fx``          | ``frac_bits``   | ``uint8_t``     | Number of       |
   |                 |                 |                 | fractional      |
   |                 |                 |                 | bits.           |
   |                 |                 |                 | Non-negative    |
   |                 |                 |                 | value.          |
   +-----------------+-----------------+-----------------+-----------------+

Kernel Specific Configuration Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Significant number of primitives must be configured by specific
parameters, which influence calculations and results, but not
directly related to input data. For example, padding and stride
values are parameters of convolution layer and the type of ReLU is a
parameter for ReLU transform layer. All specific parameters for
particular primitive type are grouped into structures. This document
describes these structures along with the primitive description they
relate to.

.. _data_muldim:
   
Data Layout of Multidimensional Feature Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions of convolution and pooling groups deal with
multi-dimensional feature maps which might be considered as images.
In general, these maps have three dimensions with following names:
height, width, and channels (also called depth). Despite logical
organization, multidimensional feature maps are stored in memory as
continuous arrays. Order of elements depends on the order of
dimensions – data layout. :ref:`Multidim_Data_Layout` describes 
two supported and traditionally used data layouts – HWC and CHW.

\

.. _Multidim_Data_Layout:
.. table:: Multidimensional Data Layout
   :widths: auto      

   +-----------------------------------+-----------------------------------+
   | Designation                       |    Description                    |
   +===================================+===================================+
   | HWC                               | The smallest stride between       |
   |                                   | dimension elements in memory is   |
   |                                   | for C (channel or depth) followed |
   |                                   | by the width and the height. The  |
   |                                   | height is the least frequently    |
   |                                   | changing index. For example       |
   |                                   | *In[32][16][8]* for this case is  |
   |                                   | a feature map with 32 rows        |
   |                                   | (**h** eight), 16 columns         |
   |                                   | (**w** idth) and 8                |
   |                                   | **c** hannels.                    |
   +-----------------------------------+-----------------------------------+
   | CHW                               | The smallest stride between       |
   |                                   | dimension elements in memory is   |
   |                                   | for W (width). Then height. The   |
   |                                   | channel is least frequently       |
   |                                   | changing index. For example       |
   |                                   | *In[32][16][8]* for this case is  |
   |                                   | a feature map with 32             |
   |                                   | **c** hannels, 16 rows            |
   |                                   | (**h** eight) and 8 columns       |
   |                                   | (**w** idth).                     |
   +-----------------------------------+-----------------------------------+

..

Due to algorithmic reasons, HWC layout provides higher data locality
for some functions, and CHW layout does so for others.

“Data locality” means the data disposition in memory where elements
consistently used by algorithm are stored in memory as close as
possible (in ideal case, contiguously). Low data locality could
reduce performance of systems with cache. For the layouts supported
by particular kernels, see :ref:`fns`. Current version of MLI Library
focuses on optimization of kernels for CHW layout.


