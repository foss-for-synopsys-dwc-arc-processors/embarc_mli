Description
-----------

   *This chapter gives a brief introduction to Machine Learning
   Inference Library API, the arithmetic involved, the package contents
   and the build process.*
   
\

General Considerations
----------------------

   The Machine Learning Inference Library is the basis for machine
   learning inference for lower power families of ARCv2 DSP cores (ARC
   EMxD and ARC HS4xD). Its purpose is to enable porting of machine
   learning models mostly based on NN to ARC processors.

   The library is a collection of ML algorithms (primitives) which
   roughly can be separated into the following groups:

   -  **Convolution** - convolve input features with a set of trained weights.

   -  **Pooling** – pool input features with a function.

   -  **Common** - Common ML, mathematical, and statistical operations

   -  **Transform** - Transform each element of input set according to a particular function

   -  **Elementwise** - Apply multi operand function element-wise to several inputs

   -  **Data manipulation** - Move input data by a specified pattern

\

   MLI supported primitives are intended for

   -  ease of use and

   -  inferring efficient solutions for small/middle models using very limited resources.
      
\
  
.. _gen_api_struct:

.. _General_API_Structure:
   
General API Structure
~~~~~~~~~~~~~~~~~~~~~

   Library is implemented as a set of C functions. Each function implements one specific NN primitive. Inputs and outputs of functions are represented by tensors. In this calculation model, graph nodes are implemented by library functions (primitives), and graph edges are represented by tensors. As a result, neural network graph implementation can be represented as series of function calls. Functions are divided on two sets:

   -  **Kernel functions**: Main implementations of ML primitives. Kernel functions process data without re-ordering to more convenient layout by copying to intermediate buffer. Avoiding such overhead provides high efficiency from one side and high sensitivity to memory latency, data size and data layout on the other side. For hardware configurations with XY memory, some kernel functions assume that data is stored in a memory that can be accessed by AGU.

   -  **Helper functions**: Provide specific functions used by primitives or required for some specific actions not directly related to graph calculations (example: data format transformation). 

\

.. _data_types:
	  
Data Types
~~~~~~~~~~

   The library is intended to work with structured data types. Each data type groups related parameters in one entity as defined in .\ **/include/MLI_types.h** .



.. _mli_tns_struct:
   
mli_tensor Structure
^^^^^^^^^^^^^^^^^^^^^

   ``mli_tensor`` is the main container type for input and output data which
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

   typedef struct \_mli_tensor {
      void *data;
      uint32_t capacity;
      uint32_t shape[MLI_MAX_RANK];
      uint32_t rank;
      mli_element_type el_type;
      mli_element_params el_params;
   } mli_tensor;

\
   
.. _mli_tensor_struc:
.. table:: mli_tensor Structure Fields
   :widths: auto
   
   +-----------------------+-----------------------+-----------------------+
   | Field name            | Field Type            | Field Description     |
   +=======================+=======================+=======================+
   |    ``data``           |    ``void \*``        | Pointer to memory     |
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

   -  Tensor shape declares more data than tensors capacity.

   -  Tensor rank is bigger than MLI_MAX_RANK

   -  Data pointer is invalid (NULL pointer) except cases when it is
      allowed to pass scalar value directly in data field.

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
   
   typedef union \_mli_element_params {
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
   |                 |                 |                 | Description     |
   | Interpretation  |                 |                 |                 |
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
   dimensions – data layout. :numref:`Multidim_Data_Layout` describes 
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
   |                                   | (**h**\ eight), 16 columns        |
   |                                   | (**w**\ idth) and 8               |
   |                                   | **c**\ hannels.                   |
   +-----------------------------------+-----------------------------------+
   | CHW                               | The smallest stride between       |
   |                                   | dimension elements in memory is   |
   |                                   | for W (width). Then height. The   |
   |                                   | channel is least frequently       |
   |                                   | changing index. For example       |
   |                                   | *In[32][16][8]* for this case is  |
   |                                   | a feature map with 32             |
   |                                   | **c**\ hannels, 16 rows           |
   |                                   | (**h**\ eight) and 8 columns      |
   |                                   | (**w**\ idth).                    |
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
   
.. _fns:
 
Functions 
~~~~~~~~~

   In general, several functions are implemented for each primitive
   supported by MLI library. Each function (implementation of primitive)
   is designed to deal with specific inputs. Therefore, you must meet the
   assumptions that functions make. For example, function designed to
   perform 2D convolution for data of ``fx8`` type must not be used with
   data of ``fx16`` type.

   All assumptions are reflected in function name according to naming
   convention (see :numref:`MLI_func_naming_conv` and 
   :numref:`MLI_fn_spl`). MLI Library functions have at
   least one assumption on input data types. Functions with only
   data-type assumption are referred to as generic functions while
   functions with additional assumptions referred to as specialized
   functions or specializations.

   .. note::    
	  A lot of specializations along with generic functions are implemented in convolution and pooling groups for each primitive. Generic functions are typically slower than the specialized ones. For this reason, a function without postfix performs switching logic to choose the correct specialized function or a generic function if there is no appropriate specialization. Such ‘switchers’ significantly increase the code size of application and should be used only in development or intentionally. Generic functions have a ‘_generic’ name postfix, and specializations have a descriptive postfix.

Naming Convention
^^^^^^^^^^^^^^^^^

   MLI Library function adheres naming convention listed in :numref:`MLI_func_naming_conv`:

\
  
.. _MLI_func_naming_conv:
.. table:: MLI Library Functions Naming Convention
   :widths: auto   

   +-----------------------+-----------------------+---------------------------------+
   | ``mli_<set>_<type>_[layout]_<data_type>_[spec](<in_data>,[config],<out_data>);``| 
   +=======================+=======================+=================================+
   | Field name            | Field Entries         | Field Description               |
   +-----------------------+-----------------------+---------------------------------+
   | ``set``               | ``krn``               | Mandatory. Specifies            |
   |                       |                       | set of functions                |
   |                       | ``hlp``               | related to the                  |
   |                       |                       | implementation. See             |
   |                       |                       | :ref:`gen_api_struct`           |
   |                       |                       | for more information.           |
   +-----------------------+-----------------------+---------------------------------+
   | ``type``              | ``conv2d``            | Mandatory. Specifies            |
   |                       |                       | particular type of              |
   |                       | ``fully_connected``   | primitive supported             |
   |                       |                       | by the library                  |
   +-----------------------+-----------------------+---------------------------------+
   | ``layout``            | ``chw``               | Optional. Specifies             |
   |                       |                       | data layout for                 |
   |                       | ``hwc``               | image-like inputs.              |
   |                       |                       | See :ref:`data_types` for       |
   |                       |                       | more information.               |
   +-----------------------+-----------------------+---------------------------------+
   | ``data_type``         | ``fx8``               | Mandatory. Specifies            |
   |                       |                       | the tensor basic                |
   |                       | ``fx16``              | element type expected           |
   |                       |                       | by the function.                |
   |                       | ``fx8w16d``           |                                 |
   |                       |                       | fx8w16d means weights           |
   |                       |                       | and bias tensors are            |
   |                       |                       | 8-bit, while all the            |
   |                       |                       | others are 16-bit.              |
   |                       |                       |                                 |
   |                       |                       | For more information,           |
   |                       |                       | see :ref:`mli_fpd_fmt`          |
   +-----------------------+-----------------------+---------------------------------+
   | ``spec``              |                       | Optional. Reflects              |
   |                       |                       | additional                      |
   |                       |                       | assumptions of                  |
   |                       |                       | function. For                   |
   |                       |                       | example, if the                 |
   |                       |                       | function can only               |
   |                       |                       | process convolutions            |
   |                       |                       | of a 3x3 kernel, this           |
   |                       |                       | should be reflected             |
   |                       |                       | in this field (see              |
   |                       |                       | :numref:`MLI_fn_spl`)           |
   +-----------------------+-----------------------+---------------------------------+
   | ``in_data``           |                       | Mandatory. Input data           |
   |                       |                       | tensors                         |
   +-----------------------+-----------------------+---------------------------------+
   | ``config``            |                       | Optional. Structure             |
   |                       |                       | of primitive-specific           |
   |                       |                       | parameters                      |
   +-----------------------+-----------------------+---------------------------------+
   | ``out_data``          |                       | Mandatory. Output               |
   |                       |                       | data tensors                    |
   +-----------------------+-----------------------+---------------------------------+

..

   Example:

   ``mli_krn_avepool_hwc_fx8(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out);``

.. _spec_fns:

Specialized Functions
^^^^^^^^^^^^^^^^^^^^^

   Naming convention for the specializations: \

.. _MLI_fn_spl:
.. table:: MLI Library Functions Naming \- Specialization Details
   :widths: auto  

   +-----------------------+---------------------------+-----------------------+
   | Configuration         |    Naming convention      | Relevant for          |
   | parameter             |                           |                       |
   +=======================+===========================+=======================+
   | ``Kernel size``       | [_k\ *n*\ x\ *m*]         | convolution group,    |
   |                       |                           | pooling group         |
   |                       | where *n* and *m* are     |                       |
   |                       | the kernel dimensions     |                       |
   |                       | example: \_k1x1, \_k3x3.  |                       |
   |                       | One of dimension might    |                       |
   |                       | be left unfixed example   |                       |
   |                       | \_k1xn                    |                       |
   +-----------------------+---------------------------+-----------------------+
   | ``Padding``           | [_nopad \| \_krnpad]      | convolution group,    |
   |                       |                           | pooling group         |
   |                       | Where \_nopad             |                       |
   |                       | functions assumes         |                       |
   |                       | that all padding          |                       |
   |                       | parameters are            |                       |
   |                       | zeros, and \_krnpad       |                       |
   |                       | functions assumes         |                       |
   |                       | smallest padding          |                       |
   |                       | parameters to achieve     |                       |
   |                       | same output size          |                       |
   |                       | (similar to ‘SAME’        |                       |
   |                       | padding scheme used       |                       |
   |                       | in TensorFlow [3])        |                       |
   +-----------------------+---------------------------+-----------------------+
   | ``Input channels``    | [_ch\ *n*]                | convolution group,    |
   |                       |                           | pooling group         |
   |                       | where *n* is the          |                       |
   |                       | number of channels        |                       |
   |                       | example \_ch1, \_ch4      |                       |
   +-----------------------+---------------------------+-----------------------+
   | ``Stride``            | [_str[h|w]\ *n*]          | convolution group,    |
   |                       |                           | pooling group         |
   |                       | where n is the stride     |                       |
   |                       | value, if needed h or     |                       |
   |                       | w can be used if          |                       |
   |                       | horizontal stride is      |                       |
   |                       | different from            |                       |
   |                       | vertical if omitted,      |                       |
   |                       | both strides are          |                       |
   |                       | equal. Example: \_str1,   |                       |
   |                       | \_strh2_strw1             |                       |
   +-----------------------+---------------------------+-----------------------+
   | ``Generalization``    | [_generic]                | convolution group,    |
   |                       |                           | pooling group         |
   |                       | If there are a lot of     |                       |
   |                       | specializations for a     |                       |
   |                       | primitive, \_generic      |                       |
   |                       | functions can process     |                       |
   |                       | inputs with any           |                       |
   |                       | combinations of           |                       |
   |                       | parameters.               |                       |
   |                       | Unspecialized             |                       |
   |                       | functions (without        |                       |
   |                       | [_spec] field in          |                       |
   |                       | name) behave as           |                       |
   |                       | “switches” which          |                       |
   |                       | analyze inputs and        |                       |
   |                       | choose suitable           |                       |
   |                       | specialization.           |                       |
   |                       | Switch   chooses          |                       |
   |                       | \_generic version in      |                       |
   |                       | case there are no         |                       |
   |                       | suitable                  |                       |
   |                       | specializations.          |                       |
   +-----------------------+---------------------------+-----------------------+

\

   For example, the function name of a 16bit 2d convolution kernel with
   CHW layout and a kernel size of 3x3 and stride of 1 is:
   ``mli_krn_conv2d_chw_fx16_k3x3_str1()``.

.. _err_codes:

Error Codes
~~~~~~~~~~~

   Functions return value of *mli_status* enumeration type which is
   declared in **include/MLI_types.h**. By default, functions do not
   validate inputs and typically return only ``MLI_STATUS_OK``.

   To turn on the checking logic, ensure that you build the MLI library
   with along with the required debug mode as described in section
   :ref:`func_param_dbg`. This might slightly affect the performance and code size of the library.

   :numref:`mli_status_val_desc` contains list of status code with description.

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
   |                                   | supported.                        |
   +-----------------------------------+-----------------------------------+
   | ``MLI_STATUS_SPEC_PARAM_MISMATCH``| Function parameters do not match  |
   |                                   | the one specified in the          |
   |                                   | specialized function.             |
   +-----------------------------------+-----------------------------------+

Global Definitions and Library Configurability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   All configurable global definitions and constants are defined in ``./include/MLI_config.h``. This header file is not included in ``./include/MLI_API.h`` header and should be included implicitly in user code in case its content might be useful. For example, use ``ARC_PLATFORM`` define for multi-platform applications.

.. _tgt_pf_def:

Target Platform Definition (ARC_PLATFORM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   ARC_PLATFORM defines main platform type that the library is built
   for. By default this is determined in compile time according to the
   TCF file. To explicitly set platform, set ARC_PLATFORM to one of the
   following macros in advance:

   -  **V2DSP** – using ARCv2DSP ISA extensions only (EM5D or EM7D).

   -  **V2DSP_WIDE** – using wide ARCv2DSP ISA extensions (HS45D or HS47D)

   -  **V2DSP_XY** – using ARCv2DSP ISA extensions and AGU (EM9D or EM11D).

.. _func_param_dbg:
   
Function Parameters Examination and Debug (MLI_DEBUG_MODE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   MLI Library supports five debug modes. You can choose the debug mode
   by setting MLI_DEBUG_MODE define as follows:

   -  **DBG_MODE_RELEASE** (**MLI_DEBUG_MODE** = 0) - No debug. Functions
      do not examine parameters. Data is processed with assumption that function 
      input is  valid.
      This might lead to undefined behavior if the assumption is not true.
      Functions always return MLI_STATUS_OK. No messages are printed, and
      no assertions are used.

   -  **DBG_MODE_RET_CODES** (**MLI_DEBUG_MODE** = 1) – Functions examine
      parameters and return valid error status if any violation of data is
      found. Else, functions process data and return status MLI_STATUS_OK.
      No messages are printed and no assertions are used.

   -  **DBG_MODE_ASSERT** (**MLI_DEBUG_MODE** = 2) - Functions examine
      parameters. If any violation of data is found, the function tries to
      break the execution using **assert()** function. If the **assert()**
      function does not break the execution, function returns error status.

   -  **DBG_MODE_DEBUG** (**MLI_DEBUG_MODE** = 3) - Functions examine
      parameters. If any violation of data is found, the function prints a
      descriptive message using standard **printf()** function and tries to
      break the execution using **assert()** function. If the **assert()**
      function does not break the execution, function returns error status.

   -  **DBG_MODE_FULL** (**MLI_DEBUG_MODE** = 4) - Functions examine
      parameters. If any violation of data is found, the function prints a
      descriptive message using standard **printf()** function and tries to
      break the execution using **assert()** function. Extra assertions inside 
      loops are used for this mode . If the **assert()**  function does not 
      break the execution, function returns error status.

   By default, ``MLI_DEBUG_MODE`` is set to ``DBG_MODE_RELEASE``.

Concatenation Primitive: Maximum Tensors to Concatenate (MLI_CONCAT_MAX_TENSORS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   This primitive configures maximum number of tensors for concatenation
   by appropriate primitive (see :ref:`concat` ). Default: 8.

Memory Allocation
~~~~~~~~~~~~~~~~~

   Library does not allocate any memory dynamically. Application is
   responsible for providing correct parameters for function and
   allocate memory for it if necessary. Library might use internal
   statically allocated data (tables of constants).

.. _hw_comp_dpd:   
   
Hardware Components Dependencies 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DSP Control
^^^^^^^^^^^

   MLI Library intensively uses ARCv2DSP extension of ARC EM and ARC HS
   processors. Ensure that this extension is present and correctly
   configured in hardware.

   Ensure that you build the library with the appropriate command line
   parameter:

   ``-Xdsp_ctrl=postshift,guard,convergent``

..   
   
   Where “up” defines the rounding mode of DSP hardware (rounding up)
   and it is the only parameter which might be changed (to “convergent” -
   round to the nearest even). All parameters are described in *MetaWare
   Fixed-Point Reference for ARC EM and ARC HS*.

.. note::
   MLI Library sets the required DSP   mode inside each function where it is needed, but does not restore it to previous state. If another ARC DSP code beside MLI library is used in an application, ensure that you set the required DSP mode before its execution. For more information see  “Configuring the ARC DSP Extensions” section of entry [4] of :ref:`refs` or “Using the FXAPI” section of entry [5] of :ref:`refs`.

AGU Support
^^^^^^^^^^^

   Library is optimized for systems with and without AGU (address
   generation unit). If AGU is present in the system, then library code
   optimized for AGU is compiled automatically, otherwise the AGU 
   optimization is not used (see :ref:`tgt_pf_def`).
   Inside primitives, pointers to some data defined with use of
   MLI_PTR(p) macro expand into “__xy p \*” in AGU systems, and to “p
   \*” in system without AGU. An application is responsible for
   allocation of relevant buffers in the AGU memory region (for more
   information see “XY Memory Optimization” chapter *of MetaWare DSP
   Programming Guide for ARC EM and ARC HS*). 

   :numref:`AGU_Req_tensors` provides information about tensors must 
   be allocated into AGUaccessible memory for each primitive. Tensors 
   not mentioned in :numref:`AGU_Req_tensors` does not have to be allocated in the 
   same way.
   
.. _AGU_Req_tensors:
.. table:: AGU Requirements for Tensors
   :widths: auto

   +-----------------------------------+-----------------------------------+
   |    Primitive                      |    Tensors must be allocated into |
   |                                   |    AGU accessible memory          |
   +===================================+===================================+
   |    Convolution 2D                 |    in, weights, out, biases       |
   +-----------------------------------+-----------------------------------+
   |    Depthwise convolution          |    in, weights, out, biases       |
   +-----------------------------------+-----------------------------------+
   |    Max Pooling                    |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Average Pooling                |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Fully connected                |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Long Short Term Memory         |    In, weights, biases, out,      |
   |                                   |    prev_out, ir_tsr               |
   +-----------------------------------+-----------------------------------+
   |    Basic RNN cell                 |    In, weights, biases, out,      |
   |                                   |    prev_out, ir_tsr               |
   +-----------------------------------+-----------------------------------+
   |    ReLU                           |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Leaky ReLU                     |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Sigmoid                        |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    TanH                           |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |    Softmax                        |    In, out                        |
   +-----------------------------------+-----------------------------------+
   |   Eltwise                         |    In1, in2, out                  |
   |   add/subtract/max/multiplication |                                   |
   |                                   |                                   |
   +-----------------------------------+-----------------------------------+
   |    Concatenation                  |    -                              |
   +-----------------------------------+-----------------------------------+
   |    Permute                        |    -                              |
   +-----------------------------------+-----------------------------------+
   |    Padding 2D                     |    -                              |
   +-----------------------------------+-----------------------------------+

\