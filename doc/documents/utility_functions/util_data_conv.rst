.. _mli_convert:

Data Conversion Group
---------------------

Description
^^^^^^^^^^^

This group contains functions which copy elements from the input tensor to the 
output with data conversion according to the output tensor type parameters. The 
functions convert the data from the source type to the destination type, 
considering the quantization parameters of the source and destinations. This 
includes a combination of:

 - Change of container size

 - Change of number of fractional bits or scale factor

 - Change of zero offset

Change of quantization axis of a per-axis quantized tensor is not allowed. 
It means that if the input tensor is quantized per specific axis, then the output tensor
must have per-tensor quantization granularity or must be quantized across the same axis 
as the input tensor.

The conversion function only supports the conversion between the MLI formats and 
the conversion formats as listed in Figure :ref:`f_data_conv_fmts`.
 
.. _f_data_conv_fmts:  
.. figure::  ../images/data_conv_fmts.png
   :align: center
   :alt: Data Conversion Formats

   Data Conversion Formats
..

Rounding or saturation are used where needed to fit the data into the destination 
container. Each output element is calculated from the corresponding input element in 
the following way:
   
.. math:: x_{\text{dst}} = Sat\left( Round \left( \left( x_{\text{src}} - z_{\text{src}} \right)*\frac{s_{\text{src}}}{2^{n_{\text{src}}}}*\frac{2^{n_{\text{dst}}}}{s_{\text{dst}}} + z_{\text{dst}}\  \right) \right)

Where:

   :math:`x_{dst},x_{src}` - destination and source sample

   :math:`z_{dst},z_{src}` = destination and source zeropoint (= 0
   for formats other than signed asymmetric)

   :math:`s_{dst},s_{src}` = destination and source scale factor (=
   1 for formats other than signed asymmetric)

   :math:`n_{dst},n_{src}` = destination and source number of
   fractional bits (= 0 for float formats)

   :math:`Sat` = saturation according to destination container size

   :math:`Round` = rounding according to destination container size

Functions
^^^^^^^^^

Function prototype:

.. code:: c

   mli_status mli_hlp_convert_tensor(
       const mli_tensor* src,
       mli_tensor* dst);
..
   
For some platforms, there is a code size penalty if floating point operations are used. 
Because the preceding function uses floating point operations even if no conversion from/to 
float is needed by the application, the linker includes the floating point support code. 
For that reason, the MLI API also includes a conversion specialization which only supports 
fixed-point data types.  

Ensure that you use this function in all places where it is known that neither the source nor 
the destination tensor is a float tensor. It supports both signed asymmetric data formats and 
fixed point data formats.

.. code:: c

   mli_status mli_hlp_convert_tensor_safx(
       const mli_tensor* src,
       mli_tensor* dst);
..
  
Conditions
^^^^^^^^^^

Ensure that you satisfy the following general conditions before calling the function:

 - ``in`` and ``out`` tensors must be valid (see :ref:`mli_tnsr_struc`)
   and satisfy data requirements of the selected version of the kernel.

 - ``in`` and ``out`` tensors must be of the same shapes

 - ``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

For **sa8** versions of the kernel, in addition to general conditions, ensure that you satisfy 
the following quantization conditions before calling the function:

 - if ``in`` and ``out`` tensors are both quantized on per-axis level, 
   then they must share the same quantization axis (``in.el_params.sa.dim`` = ``out.el_params.sa.dim``).

Result
^^^^^^

These functions only modify the memory pointed by ``out.data.mem`` field. 
It is assumed that all the other fields of ``out`` tensor are properly populated 
to be used in calculations and are not modified by the kernel.

The kernel supports in-place computation for conversions with equal container size 
(``fx8`` to/from ``sa8``, ``sa8`` to ``sa8``, ``fx16`` to ``fx16`` and etc).
It means that ``out`` and ``in`` tensor structures 
can point to the same memory with the same memory strides but without shift.
It can affect performance for some platforms.

.. warning::

  Only an exact overlap of starting address and memory stride of the ``in`` and ``out`` 
  tensors is acceptable. Partial overlaps or in-place changing of container size 
  (``fx8`` to ``fx16`` for example) result in undefined behavior.
..

Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.