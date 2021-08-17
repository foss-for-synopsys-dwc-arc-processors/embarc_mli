ARC VPX Specific Details
-------------------------

The ARC VPX family of processors combines the ARCv2 baseline ISA with ARCv2 Vector DSP ISA extension.
The latter one is actively used in MLI Library implementation for this family of processors, 
allowing us to achieve high efficiency.

 - :ref:`vpx_mem_alloc`
 - :ref:`vpx_mem_allign`
 - :ref:`vpx_accum`
 - :ref:`vpx_op_limits_shift`


.. _vpx_mem_alloc:

VPX Memory Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of almost all kernels uses vector instructions and assumes presence of operands
in the vector memory (VCCM). Which means that:

 - A memory location reference by a data container of all input and output tensors must be allocated 
   within VCCM memory region. 

 - Memory pointed to by data container of the ``mli_lut`` structure must be allocated within 
   VCCM memory region.

 - Tensors structures, LUT structures, configuration structures and memory pointed to
   by containers inside ``el_params`` field of a tensor may be allocated within any memory region. 

This applies to:
 - All functions from kernels group (see :ref:`mli_kernels`)
 - All functions related to conversion group (see :ref:`mli_convert`)

This doesn't apply to:
 - All functions from helpers group (see :ref:`mli_helpers`)
 - All functions from move group (see :ref:`data_mvmt`)

.. _vpx_mem_allign:

VPX Memory Allignement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Addresses of all elements including data, quantization parameters and structure fields 
must be aligned on an element boundary. This is also applicable for data allocated in the
vector memory (VCCM). Addresses of vectors and vector elements must be properly aligned 
on a vector-element boundary.

.. important::

   There is one type of memory access that has 8-bit alignment: a unit-stride vector load or store 
   with 8-bit elements (``fx8`` and ``sa8`` data). For the best performance vector load 
   and store access for such data must use even byte addresses (aligned on 16-bit boundary). 
   This can be achieved by using even shapes or memstrides for ``sa8`` and ``fx8`` tensors. 
   Odd byte addresses are allowed but less efficient.

..



.. _vpx_accum:

Accumulator 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The accumulator width used in calculations depends on the ``Xvec_guard_bit_option`` 
HW configuration parameter. See :ref:`quant_accum_infl` section for more info on how 
it influence the usage of the library. The following table summaries available options an
d how much accumulations it allows to do without overflow.

.. table:: VPX HW Accumulator width
   :align: center

   +-------------------+---------------+---------------------------+---------------------------+---------------------------+
   | **Kernel Type**   |               | **guard bit option = 2**  | **guard bit option = 1**  | **guard bit option = 0**  |
   +===================+===============+===========================+===========================+===========================+
   | ``sa8``           | Accum width   |     24 (8 guard bits)     |    20 (4 guard bits)      |     16 (0 guard bits)     |
   |                   +---------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs w/o      |                           |                           |                           |
   |                   | overflow      |           256             |            16             |           1               |
   |                   | guaranty      |                           |                           |                           |
   +-------------------+---------------+---------------------------+---------------------------+---------------------------+
   | ``fx16``          | Accum width   |     40 (8 guard bits)     |    36 (4 guard bits)      |     32 (0 guard bits)     |
   |                   +---------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs guaranty |                           |                           |                           |
   |                   |               |           256             |            16             |           1               |
   |                   |               |                           |                           |                           |
   +-------------------+---------------+---------------------------+---------------------------+---------------------------+
   | ``fx16_fx8_fx8``  | Accum width   |     40 (16 guard bits)    |    36 (12 guard bits)     |     32 (8 guard bits)     |
   |                   +---------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs guaranty |                           |                           |                           |
   |                   |               |           65536           |            4096           |           256             |
   |                   |               |                           |                           |                           |
   +-------------------+---------------+---------------------------+---------------------------+---------------------------+

     
..

.. _vpx_op_limits_shift:

Operands Limitations and Shifting Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

