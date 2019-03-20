.. _gen_api_struct:

Library Model
=============

   Library is implemented as a set of C functions. Each function implements one specific NN primitive. Inputs and outputs of functions are represented by tensors. In this calculation model, graph nodes are implemented by library functions (primitives), and graph edges are represented by tensors. As a result, neural network graph implementation can be represented as series of function calls. Functions are divided on two sets:

   - **Kernel functions**: Main implementations of ML primitives. Kernel functions process data without re-ordering to more convenient layout by copying to intermediate buffer. Avoiding such overhead provides high efficiency from one side and high sensitivity to memory latency, data size and data layout on the other side. For hardware configurations with XY memory, some kernel functions assume that data is stored in a memory that can be accessed by AGU.

   \
   
   - **Helper functions**: Provide specific functions used by primitives or required for some specific actions not directly related to graph calculations (example: data format transformation).

.. toctree::
   :maxdepth: 2
   
   data.rst
   error_handling.rst
   functions.rst
   hw_dependencies_config.rst