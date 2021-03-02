.. _overview: 

Overview
========

Introduction
------------

The Machine Learning Inference Library is the basis for machine learning inference for 
ARC VPX, ARC EMxD, and ARC HS4xD families of ARC DSP cores. Its purpose is to 
enable porting of machine learning models mostly based on NN to ARC processors.

The MLI library is a collection of ML algorithms (primitives) which roughly can be 
separated into the following groups: 

  - :ref:`chap_conv`

  - :ref:`chap_rec_full`

  - :ref:`chap_pool`

  - :ref:`chap_diverse_kern`

  - :ref:`chap_transform`

  - :ref:`chap_element_wise`
  
  - :ref:`chap_utlity_func`
 
MLI supported primitives are intended for

   - ease of use
   
   - inferring efficient solutions for small/mid-sized models using very limited resources 

Purpose of This Document
------------------------

This document guides you to create a MLI Implementations that is compliant to the 
desired target.

The document describes 
  
  - the usage of MLI primitives 
  
  - the technical details of MLI primitives such as 
    
    - data formats 

    - data layouts

    - quantization schemes 

    - kernel naming convention

    - supported kernels 
   
Memory Hierarchy
----------------

The functions in the MLI API operate on the lowest level in the memory hierarchy: the local 
memories (eg: DCCM, VCCM, and so on). All the memory movement needs to happen at a level above the MLI API. 
The data move functions can be used for the purpose. Inside the MLI kernels, there is no data movement. 
The strategy on data movement can be different for each platform or each application. For some 
platforms and some applications, all the data could fit in CCM, in which case no data movement 
is needed. Some platforms have a data cache; in those cases the data movement is done by the 
hardware in a transparent way. The caller/user of MLI APIs is responsible for ensuring that all 
data buffers passed to the MLI functions are in the CPU's local memory prior to making the call.

The MLI library does not allocate any memory dynamically. The caller is responsible for providing 
the correct parameters and allocated memory. The MLI library might use internal statically allocated 
data.

Header Files
------------

To ensure consistency for users, the following public header files are provided with  
MLI Implementation.  These are intended to be included by the application code:
  
  - ``mli_types.h``: This file contains definitions of the various data structures defined elsewhere 
    in this document.
  
  - ``mli_config.h``: This file describes configuration-related parameters as well as platform-specific 
    data
  
  - ``mli_api.h``: This file (or sub-include files) declares the MLI API functions

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
	
   CCM
     Closely Coupled Memory

   DCCM
     Data Closely Coupled Memory	 

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
	  
   VCCM
      Vector Closely Coupled Memory   
      
   xCAM 
      Cycle Accurate Model
  
.. _Copyright:
  
Copyright
---------


   Copyright (c) 2021, Synopsys, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided 
   that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, this list of conditions and the 
   following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this 
   list of conditions and the following disclaimer in the documentation and/or other materials provided 
   with the distribution. Neither the name of the Synopsys, Inc., nor the names of its contributors may 
   be used to endorse or promote products derived from this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
   WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
   TO, PROCUREMENT OFSUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
   OF SUCH DAMAGE. NY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   For complete embARC license information, please refer to the embARC FOSS Notice.
