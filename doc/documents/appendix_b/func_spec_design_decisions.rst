Appendix B – Functional Specification Design Decisions
======================================================

Introduction
------------

This appendix discusses some of the rational used to make the decisions 
implemented in Chapter 3 MLI API Specification.

Kernel Function Design Principles
---------------------------------

Here are some principles used to guide the design of the specification:

 - Consistent, strict naming convention for functions
 
 - Use of tensor and configuration structures as function parameters to 
   simplify API and allow separation of initialization and execution phases

 - Functions use well defined error return codes

 - Support for a variety of data inputs

 - Support for a single data layout

 - All implementations to adherence to a defined coding standard for consistency

Function naming
---------------

There is a trade off to make with respect to what information is coded in the 
function name, and what is passed as parameters to the function. The advantage of 
a specific function with some of its parameters hard-coded in the function name is 
that it can be implemented in a more optimal way with respect to cycle performance 
and code size. The advantage of a generic function is that fewer functions need to 
be implemented, this looks like a reduced complexity, but in the end the same 
information needs to travel over the interface. On one end of the spectrum, for
a very generic function like mli_krn(), everything including kernel 
functionality is passed as function arguments, and decoded at runtime. The other 
end of the spectrum has specialized functions for every optimization strategy that 
could differ per platform.

In this trade off, the following aspects were considered:

 - Different kernels can have different amount of parameters
 
 - Some specializations (like data formats) are useful on all platforms

 - Some specializations (like kernel size) are platform specific

 - More specializations can still be implemented even with a generic 
   interface, but this would require an if/else tree inside the top level 
   function. The drawback is that it is hard for the compiler (and not always 
   possible) to remove the unused code branches.
