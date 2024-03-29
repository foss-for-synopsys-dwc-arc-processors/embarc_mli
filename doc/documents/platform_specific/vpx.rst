ARC VPX Specific Details
------------------------

The ARC VPX family of processors combines the ARCv2 baseline ISA with ARCv2 Vector DSP ISA extension.
The latter one is actively used in MLI Library implementation for this family of processors, 
allowing us to achieve high efficiency.

 - :ref:`vpx_mem_alloc`
 - :ref:`vpx_mem_allign`
 - :ref:`vpx_accum`
 - :ref:`vpx_op_limits_shift`


.. _vpx_mem_alloc:

VPX Memory Allocation
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^

The accumulator width used in calculations depends on the ``Xvec_guard_bit_option`` 
HW configuration parameter. See :ref:`quant_accum_infl` section for more info on how 
it influence the usage of the library. The following table summaries available options an
d how much accumulations it allows to do without overflow.

.. tabularcolumns:: |\Y{0.19}|\Y{0.15}|\Y{0.22}|\Y{0.22}|\Y{0.22}|

.. table:: VPX HW Accumulator width
   :align: center

   +-------------------+-----------------+---------------------------+---------------------------+---------------------------+
   | **Kernel Type**   | **Description** | **guard bit option = 2**  | **guard bit option = 1**  | **guard bit option = 0**  |
   +===================+=================+===========================+===========================+===========================+
   | ``sa8``           | Accum width     |     24 (8 guard bits)     |    20 (4 guard bits)      |     16 (0 guard bits)     |
   |                   +-----------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs w/o        |                           |                           |                           |
   |                   | overflow        |           256             |            16             |           1               |
   |                   | guaranty        |                           |                           |                           |
   +-------------------+-----------------+---------------------------+---------------------------+---------------------------+
   | ``fx16``          | Accum width     |     40 (8 guard bits)     |    36 (4 guard bits)      |     32 (0 guard bits)     |
   |                   +-----------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs guaranty   |         256               |            16             |           1               |
   +-------------------+-----------------+---------------------------+---------------------------+---------------------------+
   | ``fx16_fx8_fx8``  | Accum width     |     40 (16 guard bits)    |    36 (12 guard bits)     |     32 (8 guard bits)     |
   |                   +-----------------+---------------------------+---------------------------+---------------------------+
   |                   | MACs guaranty   |          65536            |           4096            |            256            |
   +-------------------+-----------------+---------------------------+---------------------------+---------------------------+
..

.. _vpx_op_limits_shift:

Operands Limitations and Shifting Ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section describes VPX specific limitations to kernels.
In this section, :math:`n_\text{tensor}` denotes the fractional bits of a tensor
and :math:`s_\text{fx,tensor}` is its scale in case of an asymmetric data type (see :ref:`data_fmts`).

Weighted Kernels
""""""""""""""""
For the following kernels:

* conv2d
* depthwise_conv2d
* transpose_conv2d
* group_conv2d
* fully_connected
* rnn_dense
* gru_cell
* lstm_cell

Firstly, to avoid negative shifts below lower-bound and
to avoid internal large shifts above upper-bound, the the following shift restrictions must be adhered to:

.. tabularcolumns:: |\Y{0.3}|\Y{0.3}|

.. table:: 
   :align: center

   +-------------------------------+---------------------------------------------------------+
   | **Kernel Type**               | **Restriction**                                         |
   +===============================+=========================================================+
   | ``fx8``                       |   :math:`0 \leq n_{in} + n_{weight} - n_{out} \leq 15`  |
   +-------------------------------+---------------------------------------------------------+
   | ``fx16`` and ``fx16_fx8_fx8`` |   :math:`0 \leq n_{in} + n_{weight} - n_{out} \leq 31`  |
   +-------------------------------+---------------------------------------------------------+
   | ``sa8_sa8_sa32``              |   No Limitations                                        |
   +-------------------------------+---------------------------------------------------------+
..

Secondly, the following restrictions relate to shifting left the bias inside an accumulator:

.. tabularcolumns:: |\Y{0.3}|\Y{0.3}|

.. table:: 
   :align: center

   +------------------------------+-----------------------------------------------------------+
   | **Kernel Type**              | **Restriction**                                           |
   +==============================+===========================================================+
   | ``fx8``                      |   :math:`0 \leq n_{in} +  n_{weight} -  n_{bias} \leq 8`  |
   +------------------------------+-----------------------------------------------------------+
   | ``fx16``                     |   :math:`0 \leq n_{in} +  n_{weight} -  n_{bias} \leq 16` |
   +------------------------------+-----------------------------------------------------------+
   | ``fx16_fx8_fx8``             |   :math:`0 \leq n_{in} +  n_{weight} -  n_{bias} \leq 24` |
   +------------------------------+-----------------------------------------------------------+
   | ``sa8_sa8_sa32``             |   No Limitations                                          |
   +------------------------------+-----------------------------------------------------------+
..


Avepool
"""""""
**FX16**

To avoid negative shifts below lower-bound and to avoid internal large shifts
above upper-bound, the in and out fraction bits must be adhered to:

.. math::

   -14 - \text{ceil}(\text{log}_2 (\text{Wk} \cdot \text{Hk})) <
   n_\text{in} - n_\text{out}
   < 16 - \text{ceil}(\text{log}_2 (\text{Wk} \cdot \text{Hk}))
..

with :math:`\text{Wk}` and  :math:`\text{Hk}` the width and height of the kernel respectively.

**SA8**

To avoid internal large shifts below lower-bound and to avoid negative shifts
above upper-bound, the in and out scale factors must be adhered to:

.. math:: 
   
   127 \cdot 2^{-15} \cdot  \text{Wk} \cdot \text{Hk} <
   \frac{s_\text{fx,in} \cdot 2^{-n_\text{in}}}
   {s_\text{fx,out} \cdot 2^{-n_\text{out}}}
   < 64 \cdot \text{Wk}  \cdot \text{Hk}
..

with :math:`\text{Wk}` and  :math:`\text{Hk}` the width and height of the kernel respectively.


RNN Dense
"""""""""
**FX16 and FX16_FX8_FX8**

.. math::

    0 \leq n_\text{in} + n_\text{weights} - n_\text{out}
..

**SA8_SA8_SA32**

.. math::

    acc\_ scale = \frac{ s_{fx,in} s_{fx,weights}}{s_{fx,out}} 2^{n_{in} + n_{weights} - n_{out}} \\

    0 < acc\_ scale \leq 2^{32 - acc\_ size - {ceil}({log}_2 {input\_ count})}
..

where :math:`acc\_ size` is the accumulator size including the guard bits.
Restriction is to avoid saturation between multiple inputs accumulators after
the scale since accumulators are scaled and added in 32 bits vectors.


Leaky and Parametric ReLU
"""""""""""""""""""""""""
To avoid an extra shift-left instruction in the inner loop,
a negative 'slope_coeff'/'alpha' tensor fractional bits is not permitted:

.. table:: 
   :align: center

   +-------------------+------------------------+-----------------------------------+
   | **Kernel**        | **Kernel Type**        | **Restriction**                   |
   +===================+========================+===================================+
   | Leaky ReLU        | ``fx8`` and ``fx16``   |   :math:`0 \leq n_{slope\_coeff}` |
   +-------------------+------------------------+-----------------------------------+
   |  Parametric ReLU  | ``fx8`` and ``fx16``   |   :math:`0 \leq n_{alpha}`        |
   +-------------------+------------------------+-----------------------------------+
..


Element-wise Add and Element-wise Sub
"""""""""""""""""""""""""""""""""""""

**FX16**

Below restriction relates to shifting both inputs such that their fractional bits align.

.. math:: 

    \text{abs}(n_\text{in1} - n_\text{in2}) \leq 15
..


.. math::

    \text{max}(n_\text{in1}, n_\text{in2}) - 31 \leq n_\text{out} \leq  \text{max}(n_\text{in1}, n_\text{in2}) + 31 
..

**SA8**

No VPX specific limitations (see :ref:`chap_element_wise` for general limitations/requirements).



