.. _Preface: 

Preface
=======

This document describes the MLI (Machine Learning Inference) API used to define the 
interface to various processor-specific machine learning Implementations. This 
document guides you to create a MLI Implementations that is compliant to the desired target.

.. table:: Revision History 
   :align: center
   :widths: 30, 30, 130
   
   +-----------------+---------------+----------------------------------------+
   | **Date**        | **Version**   |  **Change Summary**                    |
   +=================+===============+========================================+
   | 2019, Oct 2     |     2.0       | First Version                          |
   +-----------------+---------------+----------------------------------------+
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

	  
MLI Version Numbering 
---------------------

.. table:: MLI Version Numbering
   :align: center
   :widths: 30, 130, 30

   +-----------------+-------------------------------------------+-----------------+
   | **Field Name**  | **Description**                           |  **Examples**   |
   +=================+===========================================+=================+
   | MajorVer        | Fixed at 2 for this version               |     2           |
   +-----------------+-------------------------------------------+-----------------+
   | MinorVer        | MinorVer Single digit reflects the        ||   0-alpha      |
   |                 | minor version. This allows for interim    ||   0-beta       |
   |                 | releases with minor updates which still   ||   0            |
   |                 | fall within the larger 2.x umbrella.      ||   1            |
   |                 | Prerelease versions are indicated using   ||   2            |
   |                 | ‘alpha’ or ‘beta’ prefixes.               |                 |
   +-----------------+-------------------------------------------+-----------------+
..

   
   

