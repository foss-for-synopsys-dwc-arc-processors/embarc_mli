Data Conversion Group
---------------------

This group contains functions which copy elements from the input tensor to the 
output with data conversion according to the output tensor type parameters. The 
functions converts the data from the source type to the destination type, 
considering the quantization parameters of source and destinations. This 
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

Tensor shape and rank are not changed by this function. They are copied from the source 
to the destination tensor. For conversions with equal container size, in-place computation 
is permitted. Note that this could impact performance on some platforms.
``mem_stride`` of the innermost dimension must be equal to 1 for all the tensors.

If ``mem_stride`` of the output tensor is not equal to 0, the function will use these ``mem_strides``
to store the results in the output buffer. If the ``mem_stride`` is equal to 0, 
it will be computed from the input shape.

Function prototype:

.. code:: c

   mli_status mli_hlp_convert_tensor(
       const mli_tensor* src,
       mli_tensor* dst);
..
   
For some platforms, there is a code size penalty if floating point operations are used. 
Because the preceding function uses floating point operations even if no conversion from/to 
float is needed by the application, the linker links-in the float support. For that 
reason, there is also a fixed point specialization. This function should be used in all 
places where it is known that neither of the source or destination tensor is a float tensor. 
It supports both signed asymmetric data formats and fixed point data formats. 

.. code:: c

   mli_status mli_hlp_convert_tensor_safx(
       const mli_tensor* src,
       mli_tensor* dst);
..
   
Depending on the debug level (see section :ref:`err_codes`), this function performs a parameter 
check and returns the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.