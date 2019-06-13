.. _mli_fpd_fmt:   
   
MLI Fixed-Point Data Format
---------------------------

The MLI Library targets an ARCv2DSP-based platform and implies
efficient usage of its DSP Features. Hence, there is some
specificity of basic data types and arithmetical operations using it
in comparison with operations using float-point values.

Default MLI Fixed-point data format (represented by tensors of
``MLI_EL_FX_8`` and ``MLI_EL_FX_16`` element types) reflects general signed
values interpreted by typical Q notation. The following
designation is used:

-  value of *Qm.n* format have *m* bits for integer part (excluding sign bit), 
   and *n* bits for fractional part.

-  value of *Q.n* format have *n* bits for fractional part\ *.* The rest of the 
   non-sign bits are assumed to hold an integer part.

.. note::
   For more information regarding Q notation, see 
  
   - `Q Notation`_ 

   - `Q Notation tips and tricks`_

.. _Q notation: https://en.wikipedia.org/wiki/Q_(number_format)
   
.. _Q Notation tips and tricks: http://x86asm.net/articles/fixed-point-arithmetic-and-tricks/

..

Data storage
~~~~~~~~~~~~

The container of the tensor’s values is always signed two’s
complemented integer numbers: 8 bit for ``MLI_EL_FX_8`` (also referred to as ``fx8``) and   
16 bit for ``MLI_EL_FX_16`` (also referred to as ``fx16``). ``mli_tensor`` keeps only number
of fractional bits (see ``fx.frac_bits`` in :ref:`mli_el_prm_u`),
which corresponds to the second designation above.

.. admonition:: Example 
   :class: "admonition tip"

    Given 0x4000h (16384) value in 16bit container,
    
    * In Q0.15 (and Q.15) format, this represents 0.5
    * In Q1.14 (and Q.14) format, this represents 1.0
..

For more information on how to get the real value of tensor from fx,
see :ref:`data_fmt_conv`.

Number of fractional bits must be a non-negative value. The number of
fractional bits might be larger than total number of containers
significant (not-sign) bits. In this case all bits not present in the
container implied equal to sign bit.

.. admonition:: Example 
   :class: "admonition tip"

	Given 0x0020 (32) in Q.10 format,

	• For a 16-bit container (Q5.10), this represents 0.3125 real value.

	• The value also can be stored in an 8-bit container without
	  misrepresentation. Therefore, 0x20 in Q-3.10 format is equivalent to
	  0.3125 real value.
	 
	Given 0x0220 (544) in Q.10 format,

	• For 16-bit container (Q5.10), this represent 0.53125 real value.

	• The value cannot be stored in an 8-bit container in the same Q
	  format. Therefore, conversion is required.
..
 
Values originally stored in the containers with a larger number of
bits can be represented in a container with smaller number of bits
only with a certain accuracy. Hence, values originally
stored as single precision floating point numbers cannot be
accurately represented in fx16 or fx8 formats, as single-precision floating point numbers usually have 24
bits for the mantissa.

.. note::      
   Asymmetricity of signed integer types affects FX  representation. fx8 container (int8_t) holds values in range of [-128, 127] which means that FX representation of this number is also asymmetric. So for Q.7 format, this range is [-1, 1), or
   to be more precise [-1.0, 0.9921875] (excluding 1.0). Similarly, fx16 container (int16_t) holds values in range of [-32768, 32767]. For Q.15 format, the range is [-1, 0.999969482421875].           

.. _op_fx_val:
     
Operations on FX values
~~~~~~~~~~~~~~~~~~~~~~~

Arithmetical operations are performed on signed integers
according to the rules for two’s complemented integer numbers. Q
notation gives these values a different meaning and hence,
some additional operations are required.

.. _data_fmt_conv:

Data Format Conversion
^^^^^^^^^^^^^^^^^^^^^^

Conversion between real values and fx value might be performed
according to the following formula:
 

.. math:: fx\_ val\  = Round(real\_ val\ *2^{fraq\_ bits})

.. math:: dequant\_ real\_ val\  = \frac{fx\_ val\ }{{\ 2}^{fraq\_ bits}}

where:

 - :math:`\ real\_ val\ ` \- real value (might be represented as float)
 - :math:`\ dequant\_ real\_ val\ ` \- dequantized real value
 - :math:`\ fx\_ val\ ` \- FX value of the particular Q format
 - :math:`\ fraq\_ bits \ ` \- number of :math:`\ fx\_ val\ ` fractional bits
 - :math:`\ Round\ () \ ` \- rounding according to one of supported modes

:math:`\ 2^{fraq\_ bits} \ ` represents 1.0 in FX format and also might
be obtained by shifting (1 << :math:`\ fraq\_ bits \ `). Rounding mode (nearest, up,
convergence) affects only FX representation accuracy. MLI Library
uses rounding provided by ARCv2 DSP hardware (see :ref:`hw_comp_dpd` ). :math:`\ dequant\_ real\_ val\ ` might be not equal to
:math:`\ real\_ val\ ` in case of immediate forward/backward conversion
due to rounding operation (see examples 2 and 4 from the following example list).

.. admonition:: Example 
   :class: "admonition tip"

   -  Given a real value of 0.85; FX format Q.7; rounding mode nearest, the
      FX value is computed as: 
      ``Round(0.85 * (2^7)) = Round(0.85 * 128) = Round(108.8) = 109 (0x6D)``

   -  Given a Real value -1.09; FX format Q.10; rounding mode nearest, the
      FX value is computed as:
      ``Round(-1.09 * (2^10)) = Round(-1.09 * 1024) = Round (-1116.16) =  -1116 (0xFBA4)``
	  
      	  
   -  Given an FX value 5448 in Q.15 format, the real value is computed as:
      ``5448 / (2^15) = 5448 / 32768 = 0.166259765625``

   -  Given an FX value -1116 in Q.10 format, the real value is computed as:
      ``-1116 / (2^10) = -1116 / 1024 = -1.08984375``
..

Conversion between two FX formats with different number of fractional
bits requires value shifting: shift left in case of increasing number
of fractional bits, and shift right with rounding in case of
decreasing.

.. admonition:: Example 
   :class: "admonition tip"

   -  Given an FX value 0x24 in Q.8 format (0.140625), the FX value in Q.12
      format is computed as:
      ``(0x24 << (12 – 8) ) = (0x24 << 4 ) = 0x240 in Q.12 (0.140625)``
	  

   -  Given an FX value 0x24 in Q.4 format (2.25), the FX value in Q.1format
      with rounding mode 'up' is computed as:
      ``Round(0x24>>(4–1)) = Round(0x24>>3) = (0x24 + (1<<(3-1))) >> 3 = 0x28>>3 = 0x5 in Q.1(2.5)``

Addition and Subtraction
^^^^^^^^^^^^^^^^^^^^^^^^

In fixed point arithmetic, addition and subtraction are performed as
they are for general integer values but only when the input values
are in the same format. Otherwise, ensure that you convert the 
the input values to the same format before operation.

Multiplication
^^^^^^^^^^^^^^

For multiplication, input operands do not have to be of the same
format. The width of the integer part of the result is the sum of 
widths of integer parts of the opernads. The width of the fractional 
part of the result is the sum of widths of fractional parts of the operands.

.. admonition:: Example 
   :class: "admonition tip"

   Given a number x in Q4.3 format (that is, 4 bits for integer and 3 for
   fractional part) and a number y in Q5.7 format, ``x*y`` is in Q9.10
   format (4+5=9 bits for integer part and 3+7=10 for fractional part).
..

.. note::
   For particular values,            
   multiplication might result in     
   integer value (that is, no fractional
   bits required), but for general  
   case fractional part must be     
   reserved.                         
     
..

Multiplication increases number of significant bits and requires
bigger container for intermediate result. Data conversion is
necessary for saving the multiplication result to output container
that typically does not have enough bits for holding all result. So,
unlike the addition/subtraction where conversion of inputs might be
required for inputs, multiplication typically requires conversion of
result.

Division
^^^^^^^^

For division, input operands also do not have to be of the same
format. The result has a format containing the difference of bits in
the formats of input operands.

.. admonition:: Example 
   :class: "admonition tip"

   - Given a dividend ``x`` in Q16.16 format and a divisor ``y`` in Q7.10 format,
     the format of the result ``x/y`` is Q(16-7).(16-10), or Q9.6 format.

   - Given a dividend ``x`` in Q7.8 format and a divisor ``y`` in Q3.12 format, the
     format of the result ``x/y`` is in Q4.-4 format.
..

As division is implemented using integer operation, the number of
significant bits is decreased. For the second example, sum of integer
and fractional parts of output format is 4 + (-4) = 0. This means
total precision loss for output value. To avoid this situation,
conversion of dividend operand to a larger format (with more
significant bits) is required.

Accumulation
^^^^^^^^^^^^

An addition might also result in overflow if all bits of operands
are used and both operands hold the maximum (or minimum) values. It
means that an extra bit is required for this operation. But if
sum of several operands is needed(accumulation), more than one extra bit is
required to ensure that the result does not overflow. Assuming that
all operands of the same format, the number of extra bits is defined
based on the number of additions to be done:

.. math:: extra\_ bits = \operatorname{Ceil(log_2}(number\_ of\_ additions))

..

Where :math:`\text{Ceil}(x)` function rounds up :math:`x` to the smallest integer value
that is not less than :math:`x`. From notation point of view, these extra
bits are added to integer part.

.. admonition:: Example 
   :class: "admonition tip"

   For 34 values in Q3.4 format to be accumulated, the number of extra
   bits are computed as: ceil(log\ :sub:`2` 34)= ceil(5.09) = 6
   
   Result format is: Q9.4 (since 3+6=9)
..

The same logic applies for sequential Multiply-Accumulation (MAC)
operation.

ARCv2DSP Implementation Specifics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MLI Library is designed with performance as one of the
main goals. This section deals with manual model adaptation of MLI
library.

Bias for MAC-based Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^

MAC-based kernels (convolutions, fully connected, recurrent, and so on)
typically use several input tensors including input feature map,
weights and bias (constant offset). All of them might hold data of
different FX format. The number of fractional bits is used to derive
shift values for bias and output. Such kernels perform accumulator
initialization with **left pre-shifted** bias value (format cast before
addition). Hence, the number of bias fractional bits must
be less than or equal to fractional bits for the sum of inputs. This
condition is checked by primitives in debug mode. For more
information, see :ref:`err_codes`.

.. admonition:: Example 
   :class: "admonition tip"

   Given an input tensor of Q.7 format; and weights tensor of Q.3
   format, the number of its fractional bits before shift left operation
   must be less or equal to 10 (since 7+3=10) for correct bias.
..

Configurability of Output Tensors Fractional Bits 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all primitives provide possibility to configure output tensor
format – some of them derive it based on inputs or used algorithm, 
while others must be configured with required output format explicitly. 
It depends on the basic operation used by primitive:

-  Primitives based on multiplication and division deal with
   intermediate data formats (see :ref:`op_fx_val`). If the result 
   does not fit in the output container, ensure that you provide the 
   desired result format for result conversion. Typically, it
   can not be derived from inputs and primitives of this kind requires
   output format. For example, this statement is true for convolution2D
   and fully connected.


-  Primitives based on addition, subtraction, and unary operations (max,
   min, etc) use input format (at least one of them) to perform
   calculation and save result. Conversion operation in this case is not
   required.

..

   Output configurability is specified in description for each primitive.

.. _quant_acc_bit_depth: 
   
Quantization: Influence of Accumulator Bit Depth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MLI Library applies neither saturation nor post-multiplication
shift with rounding in accumulation. Saturation is performed only for
the final result of accumulation while its value is reduced to the
output format. To avoid result overflow, user is responsible for
providing inputs of correct ranges to library primitives.

Number of available bits depends on operands types:

-  **FX8 operands**: 32-bit depth accumulator is used with 1 sign bit
   and 31 significant bits. FX8 operands have 1 sign and 7 significant
   bits. Single multiplication of such operands results in 7 + 7 = 14
   significant bits for output. Thus for MAC-based kernels, 17
   accumulation bits (as 31–(7+7)=17) are available which can be used
   to perform up to 2 :sup:`17` = 131072 operations without overflow.
   For simple accumulation, 31 – 7 = 24 bits are available which
   guaranteed to perform up to 2 :sup:`24` = 16777216 operations without
   overflow.

   
-  **FX16 operands**: 40-bit depth accumulator is used with 1 sign bit
   and 39 significant bits. FX16 operands have 1 sign and 15 significant
   bits. A multiplication of such operands results in 15 + 15 = 30
   significant bits for output. For MAC-based kernels, 39 – (15+15) = 9
   accumulation bits are available, which can be used to perform up to
   2 :sup:`9` = 512 operations without overflow.
   For simple accumulation, 39 – 15 = 24 bits are available which
   perform up to 2 :sup:`24` = 16777216 operations without overflow.

   
-  **FX16 x FX8 operands**: 32-bit depth accumulator is used. For  
   MAC-based kernels, 31 – (15 + 7) = 31 - 22 = 9 accumulation bits 
   are available which can be used to perform up to 2 :sup:`9` = 512
   operations without overflow.

In general, the number of accumulations required for one output value 
calculation can be easily estimated in advance. Using this information 
you can define if the accumulator satisfies requirements or not.
  
.. note::   
   -  If the available bits are not enough, ensure that you quantize inputs
      (including weights for both the operands of MAC) while keeping some
      bits unused.

   -  To reduce the influence of quantization on result, ensure that you 
      evenly distribute these bits between operands.
..

.. admonition:: Example 
   :class: "admonition tip"

   Given fx16 operands, 2D Convolution layer with 5x5 kernel size on
   input with 64 channels, initial Input tensor format being Q.11,
   initial weights tensor format being Q.15, each output value of 
   2D convolution layer requires the following number of accumulations:

   ``kernel_height(5) * kernel_width(5) * input_channels(64) +
   bias_add(1) = 5*5*64+1=1601``

   To ensure that the result does not overflow during accumulation, the
   following number of extra bits is required:

   ``ceil(log2(1601)) = ceil(10.65) = 11``

   9 extra bits are present in 40-bit accumulator for fx16 operands. To
   ensure no overflow, distribute 11-9=2 bits between inputs and weights
   and correct number of fractional bits. 2 is an even number and it might
   be distributed equally (-1 fractional bit for each operand).

   - The new number of fractional bits in Input tensor: = 11 – 1 = 10
   - The new number of fractional bits in Weights tensor: = 15 – 1 = 14
..
  