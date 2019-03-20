.. ML_RST documentation master file, created by
   sphinx-quickstart on Fri Feb 15 10:54:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview   
======== 
   
.. _introduction:
   
Introduction
------------
   
   The Machine Learning Inference Library is the basis for machine learning inference for lower power families of ARCv2 DSP cores (ARC EMxD and ARC HS4xD). Its purpose is to enable porting of machine learning models mostly based on NN to ARC processors.

   The library is a collection of ML algorithms (primitives) which roughly can be separated into the following groups:

   - **Convolution** - convolve input features with a set of trained weights 
   - **Pooling** – pool input features with a function 
   - **Common** - Common ML, mathematical, and statistical operations
   - **Transform** - Transform each element of input set according to a particular function
   - **Elementwise** - Apply multi operand function element-wise to several inputs
   - **Data manipulation** - Move input data by a specified pattern
   
   MLI supported primitives are intended for

   - ease of use and
   - inferring efficient solutions for small/middle models using very limited resources.
   

.. _terms_defs:

Terms and Definitions
---------------------

.. glossary::
   :sorted:

   AGU
      Address Generation Unit

   ARCv2DSP
      Synopsys DesignWare® ARC® Processors family of 32-bit CPUs
	  
   ARC EMxD
      Family of 32 bit ARC processor cores. Single core, 3-step pipeline, ARCv2DSP

   ARC HS4xD
      Family of 32 bit ARC processor cores. Multi core, dual issue, 10-step pipeline, ARCv2DSP

   CHW
      Channel-Height-Width data layout

   DMA
      Direct memory Access
	  
   DSP                               
      Digital Signal Processor 

   HWC                               
      Height-Width-Channel data layout

   IoT
      Internet Of Things      

   MAC
      Multiple Accumulate

   MWDT
      MetaWare Development Tool set 

   ML                                
      Machine Learning
	  
   MLI Library  
      Machine Learning Inference Library

   MLI kernel                         
      Basic operation on tensors in ML model, provided by MLI Library as C-style API function. Typically does not imply any intermediate copying.                          

   NN                                 
      Neural Network                    

   Primitive                         
      Basic ML functionality implemented as MLI Kernel or MLI Layer (Convolution 2D, Fully connected, and so on)            

   ReLU                              
      Rectified Linear Unit            

   TCF
      Tool Configuration File. Hold information about ARC processor build configuration and extensions.                      

   Tensor                            
      Object that contains binary data and its complete description, such as dimensions, element data type, and so on.
	  
.. _copyright:

Copyright Notice
----------------

   Copyright (c) 2019, Synopsys, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
   Neither the name of the Synopsys, Inc., nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. NY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   For complete embARC license information, please refer to the embARC FOSS Notice.