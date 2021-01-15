.. _overview: 

Overview
========

Introduction
------------

  The purpose of this document is to describe the MLI 2.0 API, which can be used by 
  developers of higher-level tools (like graph mappers/compilers) and also by those 
  creating new Implementations of MLI for various target processors. 
  
.. _f_mli_impl:  
.. figure::  ../images/mli_impl.png
   :align: center
   :alt: MLI Implementations

   MLI Implementations

    
The figure :ref:`f_mli_impl` shows how different MLI-compliant Implementations for 
different processors can all be targeted using common front-end mapping tools.  
For additional background  information on the origins of MLI, see section :ref:`history_mli`.

Purpose of This Document
------------------------

  This document formally describes the MLI API in enough detail so that you 
  can use it as a guide for creating a compliant MLI Implementation for your target processor 
  or for creating higher-level ML mapping tools which target the MLI interface. 

Context of the MLI API
----------------------

  The following chapters describe the technical details of MLI data formats, data layouts, 
  quantization schemes, kernel naming convention and supported kernels. :ref:`f_mli_api` shows how 
  the MLI API is intended to exist below the various code bases which call into it.  
  This can include the Synopsys EV Graph Mapper, another framework like TensorFlow and the 
  TensorFlow Lite for Microcontrollers backend, and even user code (manual graph mapping).  
  
  Note that some systems might need kernels which don’t exist in the MLI API.  These kernels 
  can be implemented by the user and directly called by the upper-layer code.  This provides 
  the most flexibility and allows for quick additions of new kernels outside of MLI API refresh 
  cycles.  It is recommended that user-defined kernels be implemented using the same tensor struct, 
  data formats and data layout and the same naming conventions as MLI kernels (see :ref:`func_names_special`) 
  for easier inclusion in future API versions.
 
.. _f_mli_api:  
.. figure::  ../images/mli_api.png
   :align: center
   :alt: MLI API
   
   MLI API

Memory Hierarchy
----------------

  The functions in the MLI API operate on the lowest level in the memory hierarchy: the local 
  memories (eg: DCCM, VCCM, and so on). All the memory movement needs to happen at a level above the MLI API. 
  The data move functions can be used for the purpose. Inside the MLI kernels, there is no data movement. 
  The strategy on data movement can be different for each platform or each application. For some 
  platforms and some applications, all the data could fit in CCM, in which case no data movement 
  is needed. Some platforms have a data cache; in those cases the data movement is done by the 
  hardware in a transparent way.
  
  The MLI library does not allocate any memory dynamically. The caller is responsible for providing 
  the correct parameters and allocated memory. The MLI library might use internal statically allocated 
  data.

AGU support
~~~~~~~~~~~

  The MLI Library is optimized for systems with and without AGU (address generation unit). If AGU is 
  present in the system, then library code optimized for AGU is compiled automatically, otherwise the 
  AGU optimization is not used. For systems with AGU support, the caller needs to allocate the data 
  buffers of all tensors in AGU-accessible memory.

VCCM support
~~~~~~~~~~~~

For platforms with VCCM, the MLI library assumes that (some) tensors are located in VCCM. 
TODO: document which tensors should be inside VCCM

Header Files
------------

  To ensure consistency for users, the following public header files are provided with  
  MLI Implementation.  These are intended to be included by the application code:
  
  - ``mli_types.h``: This file contains definitions of the various data structures defined elsewhere 
    in this document.
  
  - ``mli_config.h``: This file describes configuration-related parameters as well as platform-specific 
    data
  
  - ``mli_api.h``: This file (or sub-include files) declares the MLI API functions

Directory Structure
-------------------

An MLI Implementation must comply to the following directory structure to ensure consistency:

.. table:: MLI Implementation Directory Structure
   :widths: auto   

   +---------------------+---------------------------------------------------+
   | **Directory**       | **Description**                                   |
   +=====================+===================================================+
   | ``/bin``            | Built MLI library and samples binaries are        |
   |                     | created here during build                         |
   +---------------------+---------------------------------------------------+
   | ``./build``         | Contains common build rules                       |
   +---------------------+---------------------------------------------------+
   | ``./include``       | Include files with API prototypes and types       |
   +---------------------+---------------------------------------------------+
   | ``./lib/src``       | Source code of MLI Library                        |
   +---------------------+---------------------------------------------------+
   | ``./lib/gen``       | Auxiliary generation scripts for LUT tables and   |
   |                     | library source code for specialized functions     |
   +---------------------+---------------------------------------------------+  
   | ``./examples``      | Source code of examples                           |
   +---------------------+---------------------------------------------------+
..   
    
.. _terms_and_defs:
   
Terms and definitions
---------------------

.. glossary::
   :sorted:

   AGU
      Address Generation Unit

   API 
      Application Programming Interface
  
   ARCv2DSP 
      Synopsys DesignWare® ARC® Processors Family of 32-bit CPUs 

   ARC EMxD 
      Family of 32-bit ARC Processor Cores. Single-core, 3-Step Pipeline, ARCv2DSP 

   ARC HS4xD 
      Family of 32-bit ARC Processor Cores. Multi-core, Dual-Issue, 10-Step Pipeline, ARCv2DSP
      
   CCAC 
      MetaWare Compiler 

   DMA 
      Direct Memory Access 
      
   DSP
      Digital Signal Processor 
      
   FXAPI 
      Fixed-point API 
     
   LTO 
      Link-Time Optimization 

   MAC
      Multiple Accumulate 

   MDB 
      MetaWare Debugger

   MPY 
      Multiply Command 

   MWDT
      MetaWare Development Toolset
      
   nSIM 
      Instruction Set Simulator
      
   OOB
      Out-Of-the Box   

   PCM 
      Pulse Code Modulation 
   
   TCF
      Tool Configuration File. Holds information about ARC processor build configuration and extensions. 
      
   xCAM 
      Cycle Accurate Model
  
    
.. _Copyright:
  
Copyright
---------

  Copyright TBD
