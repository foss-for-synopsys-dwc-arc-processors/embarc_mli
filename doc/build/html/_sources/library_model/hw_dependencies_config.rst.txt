Hardware Dependencies and Configurability
-----------------------------------------

Global Definitions and Library Configurability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All configurable global definitions and constants are defined in **./include/mli_config.h**. This header file is not included in **./include/mli_api.h** header and should be included implicitly in user code in case its content might be useful. For example, use ``ARC_PLATFORM`` define for multi-platform applications.

.. _tgt_pf_def:

Target Platform Definition (ARC_PLATFORM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ARC_PLATFORM defines main platform type that the library is built
for. By default this is determined in compile time according to the
TCF file. To explicitly set the platform, set ARC_PLATFORM to one of the
following macros in advance:

-  **V2DSP** – using ARCv2DSP ISA extensions only (EM5D or EM7D).

-  **V2DSP_WIDE** – using wide ARCv2DSP ISA extensions (HS45D or HS47D)

-  **V2DSP_XY** – using ARCv2DSP ISA extensions and AGU (EM9D or EM11D).

.. _func_param_dbg:
   
Function Parameters Examination and Debug (MLI_DEBUG_MODE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLI Library supports five debug modes. You can choose the debug mode
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
   parameters.  Any violation of data found lead to break on **assert()** 

-  **DBG_MODE_DEBUG** (**MLI_DEBUG_MODE** = 3) - The same as DBG_MODE_ASSERT, 
   but before breaking on **assert()** function prints descriptive message 
   using standard **printf()**

-  **DBG_MODE_FULL** (**MLI_DEBUG_MODE** = 4) - The same as DBG_MODE_DEBUG,
   but additionally extra assertions inside loops are used for this mode.

By default, ``MLI_DEBUG_MODE`` is set to ``DBG_MODE_RELEASE``. The following table 
specifies modes behavior.

.. _DBG_Mode_Behav:
.. table:: MLI_DEBUG_MODE modes behavior
    
   +----------------------+-----------+-------------+----------+---------+--------+
   |    Behavior / Mode   |  RELEASE  |  RET_CODES  |  ASSERT  |  DEBUG  |  FULL  |
   +======================+===========+=============+==========+=========+========+
   |    Return Codes      |   NO      |   YES       |   YES    |  YES    |  YES   |
   +----------------------+-----------+-------------+----------+---------+--------+
   |    Assertions        |   NO      |   NO        |   YES    |  YES    |  YES   |
   +----------------------+-----------+-------------+----------+---------+--------+
   |    Extra Assertions  |   NO      |   NO        |   NO     |  NO     |  YES   |
   +----------------------+-----------+-------------+----------+---------+--------+
   |    Messages          |   NO      |   NO        |   NO     |  YES    |  YES   |
   +----------------------+-----------+-------------+----------+---------+--------+



Concatenation Primitive: Maximum Tensors to Concatenate (MLI_CONCAT_MAX_TENSORS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This primitive configures maximum number of tensors for concatenation
by appropriate primitive (see :ref:`concat` ). Default: 8.

Memory Allocation
~~~~~~~~~~~~~~~~~

The MLI library does not allocate any memory dynamically. Your application is
responsible for providing correct parameters for function and
allocate memory for it if necessary. The MLI library might use internal
statically allocated data (tables of constants).

.. _hw_comp_dpd:   
   
Hardware Components Dependencies 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DSP Control
^^^^^^^^^^^

The MLI Library intensively uses ARCv2DSP extension of ARC EM and ARC HS
processors. Ensure that this extension is present and correctly
configured in hardware.

Ensure that you build the library with the appropriate command line
parameter:

``-Xdsp_ctrl=postshift,guard,convergent`` 
   
Where “up” defines the rounding mode of DSP hardware (rounding up)
and it is the only parameter which might be changed (to “convergent” -
round to the nearest even). All parameters are described in *MetaWare
Fixed-Point Reference for ARC EM and ARC HS*.

.. note::
   The MLI Library sets the required DSP mode inside each function where it is needed, but does not restore it to previous state. If another ARC DSP code beside MLI library is used in an application, ensure that you set the required DSP mode before its execution. For more information see  “Configuring the ARC DSP Extensions” section of *MetaWare DSP Programming Guide for ARC EM and ARC HS* or “Using the FXAPI” section of *MetaWare Fixed-Point Reference for ARC EM and ARC HS*.

AGU Support
^^^^^^^^^^^

The MLI Library is optimized for systems with and without AGU (address
generation unit). If AGU is present in the system, then library code
optimized for AGU is compiled automatically, otherwise the AGU 
optimization is not used (see :ref:`tgt_pf_def`).
Inside primitives, pointers to some data defined with use of
``MLI_PTR(p)`` macro expand into ``__xy p *`` in AGU systems, and to ``p
*`` in system without AGU. An application is responsible for
allocation of relevant buffers in the AGU memory region (for more
information see “XY Memory Optimization” chapter *of MetaWare DSP
Programming Guide for ARC EM and ARC HS*). 

:ref:`AGU_Req_tensors` provides information about tensors must 
be allocated into AGU-accessible memory for each primitive. Tensors 
not mentioned in :ref:`AGU_Req_tensors` do not have to be allocated in the 
same way.
   
.. _AGU_Req_tensors:
.. table:: AGU Requirements for Tensors
   :widths: 20,130

   +-----------------------------------+-----------------------------------+
   |    Primitive                      |    Tensors That Must Be Allocated |
   |                                   |    Into AGU-Accessible Memory     |
   +===================================+===================================+
   |    Convolution 2D                 |    in, weights, out, biases       |
   +-----------------------------------+-----------------------------------+
   |    Depthwise convolution          |    in, weights, out, biases       |
   +-----------------------------------+-----------------------------------+
   |    Max Pooling                    |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Average Pooling                |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Fully connected                |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Long Short Term Memory         |    in, weights, biases, out,      |
   |                                   |    prev_out, ir_tsr               |
   +-----------------------------------+-----------------------------------+
   |    Basic RNN cell                 |    in, weights, biases, out,      |
   |                                   |    prev_out, ir_tsr               |
   +-----------------------------------+-----------------------------------+
   |    ReLU                           |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Leaky ReLU                     |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Sigmoid                        |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    TanH                           |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |    Softmax                        |    in, out                        |
   +-----------------------------------+-----------------------------------+
   |   Eltwise                         |    in1, in2, out                  |
   |   add/subtract/max/multiplication |                                   |
   |                                   |                                   |
   +-----------------------------------+-----------------------------------+
   |    Concatenation                  |    -                              |
   +-----------------------------------+-----------------------------------+
   |    Permute                        |    -                              |
   +-----------------------------------+-----------------------------------+
   |    Padding 2D                     |    -                              |
   +-----------------------------------+-----------------------------------+

