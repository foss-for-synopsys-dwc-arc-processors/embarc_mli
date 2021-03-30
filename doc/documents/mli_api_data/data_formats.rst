.. _data_fmts:

Data Formats
------------

This section describes the set of possible data formats present in the MLI interface. 
Hereinafter, ‘data format’ means the way of representing individual values which are 
grouped into the tensor structure (see section :ref:`mli_tens_data_struct` ). The table
:ref:`mli_data_fmts` summarizes the data formats and the following sections describe 
these data formats in detail.  

.. note::
   The data formats in this table do NOT imply unique implementations of MLI kernels 
   for each data format.  The last column in the table describes each the purpose of 
   each data format. The available set of kernels is described in :ref:`mli_kernels`.
   
.. _mli_data_fmts:
.. table:: MLI Data Formats
   :align: center
   :widths: auto
   
   +--------------+------------+---------------+-------------+------------------+------------------------+
   | **Format**   | **Data**   | **C Type**    | **Format**  | **Quantization** | **Usage in MLI 2.0**   | 
   | **Category** | **Format** | **Container** | **Name**    | **Granularity**  |                        |
   +==============+============+===============+=============+==================+========================+
   | Fixed Point  |   fx16     | int16_t       | 16-bit      | Per-tensor       | Used as main 16-bit    |
   |              |            |               | fixed point |                  | data format for kernel |
   |              |            |               |             |                  | inputs and outputs     |
   |              |            |               |             |                  |                        |
   |              +------------+---------------+-------------+------------------+------------------------+
   |              |   fx8      | int8_t        | 8-bit       | Per-tensor       | Only in conversion     |
   |              |            |               | fixed point |                  | function and in        |
   |              |            |               |             |                  | combination with fx16  |
   +--------------+------------+---------------+-------------+------------------+------------------------+
   | Asymmetric   |   sa32     | int32_t       | 32-bit      | Per-tensor       | Used only for bias     |
   | Integral     |            |               | signed      | Per-axis         | inputs to kernels      |
   |              |            |               | symmetric   |                  |                        |
   |              +------------+---------------+-------------+------------------+------------------------+     
   |              |   sa8      | int8_t        | 8-bit       | Per-tensor       | Used as main 8-bit     |
   |              |            |               | signed      | Per-axis         | data format for kernel |
   |              |            |               | symmetric   |                  | inputs and outputs     |
   +--------------+------------+---------------+-------------+------------------+------------------------+
   | Floating     |   fp32     | float         | 32-bit      | Per-value        | Only in conversion     |
   | Point        |            |               | floating    |                  | function as interface  |
   |              |            |               | point       |                  | between MLI and user   | 
   |              |            |               |             |                  | code                   |
   +--------------+------------+---------------+-------------+------------------+------------------------+    
..                                                
                                                 
.. note::
   Quantization Granularity – The way in which quantization parameters (exponent, 
   fractional bits, and so on) are shared between values. 

   Possible entities:

   - Per-tensor: All values in tensor shares the same quantization parameters

   - Per-axis: All values in tensor across one of axis shares the same quantization 
     parameters (most typical example – per channel quantization)

   - Per-value: Each individual value is configured with its own set of quantization 
     parameters
..
                                
Fixed Point Category
~~~~~~~~~~~~~~~~~~~~

The Fixed-Point category includes ``fx16`` and ``fx8`` data formats. They are the 
default MLI Fixed-point data format and reflects general signed values interpreted 
by the typical `Q notation <https://en.wikipedia.org/wiki/Q_(number_format)>`_. 
The following designation is typically used:

  - Value of Qm.n format have m bits for integer part (excluding sign bit), and 
    n bits for fractional part.
    
  - Value of Q.n format have n bits for fractional part. The rest m non-sign bits 
    are assumed to hold an integer part. 
                                
The container of the value is always a signed two’s complemented integer number. 
Approximation of single precision floating point value and backward transformation 
are performed by the following formula:                                                  
                                                 
.. math::
   x_{fx} &= Round(x_{fp32} * 2^n)

   x_{fp32} &= \frac{x_{{fx}}}{2^{n}}
   
..

   where :math:`x_{fp32}` *-* single precision floating point value
            
         :math:`x_{fx}` *-* fixed point value
         
         :math:`n` *-* the number of fractional bits
         
         :math:`Round(\ldots)` *-* rounding to integer value
         
:math:`2^{n}` represents 1.0 in FX format and also might be obtained by shifting :math:`1 <<  n`. 
Rounding mode (nearest, up, convergence, truncated, and so on) affects only FX representation precision 
and can be platform specific. If the calculated :math:`x_{fx}` fixed point value exceeds container type 
range, it must be saturated. In case of immediate forward/backward conversion, :math:`x_{fp32}` might be 
not equal to the original one due to rounding and saturation operations. Only per-tensor 
quantization granularity is supported for these data formats, which means that all values in the 
tensor share the same quantization parameters (number of fractional bits).

An addition of two :math:`{fx}` values might result in overflow if all bits of operands are used and both 
operands hold the maximum (or minimum) values. It means that an extra bit is required for this 
operation. But if the sum of several operands is needed (accumulation), more than one extra bit is 
required to ensure that the result does not overflow. Assuming that all operands of the same 
format, the number of extra bits is defined based on the number of additions to be done:

.. math::

   extra\_ bits = Ceil({\log}_{2}(number\_ of\_ operands))         
..

Where :math:`Ceil(x)` – function rounds up x to the smallest integer value that is not less 
than x. From notation point of view, these extra bits are added to the integer part.   

.. admonition:: Example 
   :class: "admonition tip"

   For 34 values in Q3.4 format to be accumulated, the number of extra bits is computed as: 

   - :math:`\text{Ceil}(log_2 34) = ceil(5.09) = 6` 

   - Result format is: Q9.4 (since 3+6=9)
..

The same logic applies for sequential Multiply-Accumulation (MAC) operations.

Asymmetric Integral category
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Asymmetric Integral category includes ``sa32`` and ``sa8`` data formats. These data formats are used 
for more precise quantized representation of asymmetrically distributed data. To correctly 
interpret values of this data format, quantization scale ration (s) and zero offset (z) must be 
provided. Approximation of single precision floating point value and backward transformation are 
performed by:

.. math:: 

   x_{\text{sa}} = Round\left( \left( \frac{x_{fp32}}{{(s}_{\text{fx}}*2^{- n})} \right) + z \right)
   
   x_{fp32} = \left( x_{\text{sa}} - z \right)*{(s}_{\text{fx}}*2^{- n})

..

Where: 

    :math:`x_{fp32}` *–* Source single precision floating point value
    
    :math:`x_{sa}` *–* signed asymmetric value
    
    :math:`z` *–* zero offset
    
    :math:`Round(\ldots)` *–* rounding to integer value. 
    
    :math:`s_{\text{fx}}` *–* scale ratio in fixed point format
    
    :math:`n` *–* number of fractional bits of scale ratio. 
    
Per-axis and per-tensor quantization granularities are supported for this data format. In case of 
per-tensor quantization, all values in tensor share the same quantization parameters (number scale 
ratio and zero offset). In case of per-axis quantization, each slice of the tensor across a defined axis 
is configured with individual quantization parameters (scale ratio and zero offset). 

Asymmetric integral data format is a more generic and flexible representation in comparison with 
the fixed point data format. But this flexibility also implies additional complexity in calculations, 
and extra assumptions to simplify it at inference time. These assumptions are listed along with 
the description of each kernel in :ref:`mli_kernels`. 

Fixed point data format can be considered as special case of asymmetric integer data with 
assumption that  :math:`z=0` and :math:`s_{fx}=1`, which allows you to use only 
shift operations to change (requantize) data format not involving zero points and 
specific scale ratios:

.. math::

   Round\left( \left( \frac{x_{fp32}}{(s_{fx}*2^{- n})} \right) + z \right) = \ Round\left( \left( \frac{x_{fp32}}{(1*2^{- n})} \right) + 0 \right) = Round\left( x_{fp32}*2^{n} \right) = x_{{fx}}
..

.. _quant_accum_infl:

Quantization: Influence of Accumulator Bit Depth   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MLI Library applies neither saturation nor post multiplication shift with rounding in 
accumulation. Saturation is performed only for the final result of accumulation while its 
value is reduced to the output format. To avoid result overflow, you are responsible for 
providing inputs of correct ranges to MLI library primitives.

Number of available bits depends on the operands’ types and the platform. 

.. admonition:: Example 
   :class: "admonition tip"

   - ``sa8`` operands with 32-bit accumulator uses 1 sign bit and 31 significant bits. ``sa8`` operands 
     have 1 sign and 7 significant bits. Single multiplication of such operands results in 
     7 + 7 + 1 = 15 significant bits for output. Here one extra bit is required to handle multiplication 
     of max negative values (-32768 * -32768 = 1073741824 – the value of 31 bits depth). 
     Thus for MAC-based kernels, 16 accumulation bits (as 31-(7+7+1)=16) are available which can be used to
     perform up to 2^16 = 65536 operations without overflow. For simple accumulation, 31 – 7 = 24 bits are
     available which are guaranteed to perform up to 2^24 = 16777216 operations without overflow.

   - ``fx16`` operands with 40-bit accumulator is uses 1 sign bit and 39 significant bits. ``fx16`` 
     operands have 1 sign and 15 significant bits. A multiplication of such operands results in 
     15 + 15 + 1 = 31 significant bits for output. Here one extra bit is required to handle multiplication 
     of max negative values (-128 * -128 = 16384 – the value of 15 bits depth). For MAC-based kernels, 
     39 – (15+15+1) = 8 accumulation bits are available, which can be used to perform up to 2^8 = 256 
     operations without overflow. For simple accumulation, 39 – 15 = 24 bits are available which 
     perform up to 2^24 = 16777216 operations without overflow.
..

In general, the number of accumulations required for one output value calculation can be  
estimated in advance. 

.. note::

   - If the available bits are not enough, ensure that you quantize inputs (including weights for 
     both the operands of the MAC) while keeping some bits unused.
     
   - To reduce the influence of quantization on the result, ensure that you evenly distribute these bits 
     between operands.
..

Special functions to determine the number of the available accumulator guard bits for the different operand 
combination are provided. These values can be different when compiled on a different platform. 
These functions are defined in :ref:`num_of_accu_bits` section.

