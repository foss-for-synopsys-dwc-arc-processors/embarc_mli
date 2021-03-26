.. _mli_lut_data_struct:

mli_lut Data Structure
--------------------------

Several functions use a look-up table (LUT) to perform data transformation.  The LUT represents a function in a 
table form that can be used to transform input values (function argument) to output values (function result). 
``mli_lut`` structure is a representation of such table.

The ``mli_lut`` struct describes the data in the LUT, including the format of its input and output.


.. code:: c

   typedef struct _mli_lut{
      mli_data_container data;
      mli_element_type type;
      int32_t length;
      int32_t in_frac_bits;
      int32_t out_frac_bits;
      int32_t input_offset;
      int32_t output_offset;
   } mli_lut;
..

See :ref:`mli_tens_data_struct` for ``mli_data_container`` and ``mli_element_type`` structures definition. Te following table describes the fields in the mli_lut structure
   
.. _mli_lut_struct_table:  
.. table:: mli_lut Structure Field Descriptions
   :align: center
   :widths: 50, 50, 130 
   
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | **Field name**    | **type**               | **Comment**                                                                 |
   +===================+========================+=============================================================================+
   |                   |                        | This field has a union of different possible data container types.          |
   |   ``data``        | ``mli_data_container`` | Pointer of specified type (see the type field below) should point to        |
   |                   |                        | an array with the LUT table data.                                           |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``data.capacity`` | ``uint32_t``           | Size in bytes of the allocated memory that the data field points to.        |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``type``          | ``mli_element_type``   | Enum depicting the type of the element stored in the data field.            |
   |                   |                        | Values in this enum are listed in section :ref:`mli_tens_data_struct`.      |
   |                   |                        | Only ``MLI_EL_FX_8`` and ``MLI_EL_FX_16`` entities are supported.           |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``length``        | ``int32_t``            | Number of values stored in the LUT table                                    |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``in_frac_bits``  | ``int32_t``            | Number of fractional bits for LUT input (argument)                          |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``out_frac_bits`` | ``int32_t``            | Number of fractional bits for LUT output (result)                           |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``input_offset``  | ``int32_t``            | Offset of input argument which is added before applying the LUT function.   |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
   | ``output_offset`` | ``int32_t``            | Offset of output which is subtracted from LUT function result.              |
   +-------------------+------------------------+-----------------------------------------------------------------------------+
     
..
   