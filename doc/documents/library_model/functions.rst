.. _fns:

Functions 
---------

In general, several functions are implemented for each primitive
supported by MLI library. Each function (implementation of primitive)
is designed to deal with specific inputs. Therefore, you must meet the
assumptions that functions make. For example, function designed to
perform 2D convolution for data of ``fx8`` type must not be used with
data of ``fx16`` type.

All assumptions are reflected in function name according to naming
convention (see :ref:`MLI_func_naming_conv` and 
:ref:`MLI_fn_spl`). MLI Library functions have at
least one assumption on input data types. Functions with only
data-type assumption are referred to as generic functions while
functions with additional assumptions referred to as specialized
functions or specializations.

.. note::    
  A lot of specializations along with generic functions are implemented in convolution and pooling groups for each primitive. Generic functions are typically slower than the specialized ones. Hence, a function without postfix performs switching logic to choose the correct specialized function or a generic function if there is no appropriate specialization. Such ‘switchers’ significantly increase the code size of application and should be used only in development or intentionally. Generic functions have a ‘_generic’ name postfix, and specializations have a descriptive postfix.

Naming Convention
~~~~~~~~~~~~~~~~~

MLI Library function adheres naming convention listed in :ref:`MLI_func_naming_conv`:

.. code::

   mli_<set>_<type>_[layout]_<data_type>_[spec](<in_data>,[config],<out_data>) ; 
.. 
 
.. _MLI_func_naming_conv:
.. table:: MLI Library Functions Naming Convention
   :widths: 30,20,130   

   +-----------------------+-----------------------+------------------------------------------------------+
   | **Field name**        | **Field Entries**     | **Field Description**                                |
   +=======================+=======================+======================================================+
   | ``set``               | ``krn``               | Mandatory. Specifies                                 |
   |                       |                       | set of functions                                     |
   |                       | ``hlp``               | related to the                                       |
   |                       |                       | implementation. For more information, see            |
   |                       |                       | :ref:`gen_api_struct`                                |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``type``              | ``conv2d``            | Mandatory. Specifies                                 |
   |                       |                       | particular type of                                   |
   |                       | ``fully_connected``   | primitive supported                                  |
   |                       |                       | by the library                                       |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``layout``            | ``chw``               | Optional. Specifies                                  |
   |                       |                       | data layout for                                      |
   |                       | ``hwc``               | image-like inputs.                                   |
   |                       |                       | For more information, see :ref:`data_types`          |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``data_type``         | ``fx8``               | Mandatory. Specifies                                 |
   |                       |                       | the tensor basic                                     |
   |                       | ``fx16``              | element type expected                                |
   |                       |                       | by the function.                                     |
   |                       | ``fx8w16d``           | ``fx8w16d`` means weights                            |
   |                       |                       | and bias tensors are                                 |
   |                       |                       | 8-bit, while all the                                 |
   |                       |                       | others are 16-bit.                                   |
   |                       |                       | For more information,                                |
   |                       |                       | see :ref:`mli_fpd_fmt`                               |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``spec``              |                       | Optional. Reflects                                   |
   |                       |                       | additional                                           |
   |                       |                       | assumptions of                                       |
   |                       |                       | function. For                                        |
   |                       |                       | example, if the                                      |
   |                       |                       | function can only                                    |
   |                       |                       | process convolutions                                 |
   |                       |                       | of a 3x3 kernel, this                                |
   |                       |                       | should be reflected                                  |
   |                       |                       | in this field (see                                   |
   |                       |                       | :ref:`MLI_fn_spl`)                                   |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``in_data``           |                       | Mandatory. Input data          	                  |
   |                       |                       | tensors                        	                  |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``config``            |                       | Optional. Structure            	                  |
   |                       |                       | of primitive-specific          	                  |
   |                       |                       | parameters                     	                  |
   +-----------------------+-----------------------+------------------------------------------------------+
   | ``out_data``          |                       | Mandatory. Output                                    |
   |                       |                       | data tensors                                         |
   +-----------------------+-----------------------+------------------------------------------------------+

..
.. note::  

   Example:
   
   .. code::
   
      mli_krn_avepool_hwc_fx8(const mli_tensor *in, 
                              const mli_pool_cfg *cfg, 
                              mli_tensor *out
                              );
..
   
.. _spec_fns:

Specialized Functions
~~~~~~~~~~~~~~~~~~~~~

Naming convention for the specializations: \

.. _MLI_fn_spl:
.. table:: MLI Library Functions Naming \- Specialization Details
   :widths: 20,60,20  

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


For example, the function name of a 16bit 2d convolution kernel with
CHW layout and a kernel size of 3x3 and stride of 1 is:
``mli_krn_conv2d_chw_fx16_k3x3_str1()``.

