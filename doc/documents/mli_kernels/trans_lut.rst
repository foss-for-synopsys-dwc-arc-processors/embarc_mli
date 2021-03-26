.. _lut_prot:

Look-Up Tables (LUT) Manipulation Prototypes and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several MLI kernels use a look-up table (LUT) to perform data transformation. 
The LUT should be prepared by the user before the kernel call and provided as a 
kernel parameter. The LUT preparation implies the following steps:

 1. Getting memory size required for the LUT

 2. Memory allocation of defined size

 3. LUT structure creation


Functions which returns the size of the memory for the LUT table (step 1) have the following prototype:

.. code:: c

   int32_t mli_krn_<lut_name>_get_lut_size();

..

where ``lut_name`` is one of supported LUT tables.

The user is responsible for allocating memory for the LUT table and must assign it to the LUT structure 
before calling the creation function (step 2).

Functions which create a specific LUT table (step 3) have the following prototype: 

.. code:: c

   mli_status mli_krn_<lut_name>_create_lut(mli_lut *lut);

..

where ``lut_name`` is one of supported LUT tables within MLI and the function parameters are shown below:

.. table:: LUT Creation Function Parameters
   :align: center
   :widths: auto 
   
   +----------------+----------------------+---------------------------------------------------------------------+
   | **Parameter**  | **Type**             | **Description**                                                     |
   +================+======================+=====================================================================+
   | ``lut``        | ``mli_lut *``        | [IN | OUT] Pointer to the LUT table structure with assigned memory. |
   +----------------+----------------------+---------------------------------------------------------------------+
..

``mli_lut`` is defined in the :ref:`mli_lut_data_struct` section. 

Here is a list of all available LUT manipulation functions:

.. table:: List of LUT Manipulation  Functions
   :align: center
   :widths: auto 
   
   +----------------------------------------+-----------------------------------------------------------+
   | **Function Name**                      | **Details**                                               |
   +========================================+===========================================================+
   | ``mli_krn_sigm_get_lut_size``          | Get the size of the sigmoid activation LUT                |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_tanh_get_lut_size``          | Get the size of the hyperbolic tangent activation LUT     |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_softmax_get_lut_size``       | Get the size of the softmax activation LUT                |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_l2_normalize_get_lut_size``  | Get the size of the L2 Normalization LUT                  |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_sigm_create_lut``            | Create the sigmoid activation LUT                         |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_tanh_create_lut``            | Create the hyperbolic tangent activation LUT              |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_softmax_create_lut``         | Create the softmax activation LUT                         |
   +----------------------------------------+-----------------------------------------------------------+
   | ``mli_krn_l2_normalize_create_lut``    | Create the L2 Normalization LUT                           |
   +----------------------------------------+-----------------------------------------------------------+
..


There are no any specific requirements for  ``mli_krn_<lut_name>_get_lut_size`` functions. It can be called at any time.

Ensure that you satisfy the following conditions before calling the ``mli_krn_<lut_name>_create_lut`` function:

 - ``in`` tensor must be quantized on the tensor level. This implies that the tensor 
   contains a single scale factor and a single zero offset.
 - ``lut`` structure must contain a valid pointer to a buffer with sufficient capacity which was defined 
   by the corresponding ``*_ get_lut_size`` function.


Below is a pseudo-code sample of LUT manipulation functions usage together with LUT-consumer kernel call. 


.. code:: c

   // Allocate empty LUT structure. 
   // Any memory class is allowed (automatic, dynamic, static). 
   mli_lut lut;

   // Step 1: Get required LUT size and check it against pre-allocated buffer.
   int lut_size = mli_krn_<lut_name>_get_lut_size();
   assert(lut_size <= user_pre_allocated_buffer_size);
   assert(user_pre_allocated_buffer != NULL);

   // Step 2: Assign a memory region to the data of LUT structure.
   // Implementation can put extra requirements for LUT data memory.
   lut.data.mem.pi16 = (int16_t*)user_pre_allocated_buffer;
   lut.data.capacity = lut_size;

   // Step 3: create and validate LUT structure.
   mli_status ret_code = mli_krn_<lut_name>_create_lut(&lut);
   assert(ret_code == MLI_STATUS_OK);

   // A valid LUT structure can be used by the consumer kernel
   mli_krn_<lut_consumer>(..., &lut, ...);

..

