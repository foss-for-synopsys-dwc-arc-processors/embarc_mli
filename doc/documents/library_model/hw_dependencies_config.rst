Hardware Dependencies and Configurability
-----------------------------------------

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
   MLI Library sets the required DSP   mode inside each function where it is needed, but does not restore it to previous state. If another ARC DSP code beside MLI library is used in an application, ensure that you set the required DSP mode before its execution. For more information see  “Configuring the ARC DSP Extensions” section of *MetaWare DSP Programming Guide for ARC EM and ARC HS* or “Using the FXAPI” section of entry [5] of *MetaWare Fixed-Point Reference for ARC EM and ARC HS*.

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

   :ref:`AGU_Req_tensors` provides information about tensors must 
   be allocated into AGUaccessible memory for each primitive. Tensors 
   not mentioned in :ref:`AGU_Req_tensors` does not have to be allocated in the 
   same way.
   
.. _AGU_Req_tensors:
.. table:: AGU Requirements for Tensors
   :widths: grid

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
   