.. _kernl_sp_conf:

Kernel Specific Configuration Structures
----------------------------------------

A significant number of MLI kernels must be configured by specific parameters, which 
influence calculations and results, but are not directly related to input data. For 
example, padding and stride values are parameters of the convolution layer and the type 
of ReLU is a parameter for ReLU transform layer. All specific parameters for 
particular primitive type are grouped into structures. This document describes these 
structures along with the kernel description they relate to. The following tables 
describe fields of existing MLI configuration structures:

 - Table :ref:`t_mli_conv2d_cfg_desc`
 
 - Table :ref:`t_mli_fc_cfg_desc` 

 - Table :ref:`t_mli_rnn_cell_cfg_desc` 

 - Table :ref:`t_mli_rnn_dense_cfg_desc`

 - Table :ref:`t_mli_pool_cfg_desc` 

 - Table :ref:`t_mli_argmax_cfg_desc`

 - Table :ref:`t_mli_permute_cfg_desc`

 - Table :ref:`t_mli_relu_cfg_desc`

 - Table :ref:`t_mli_prelu_cfg_desc`

 - Table :ref:`t_mli_mov_cfg_desc`

..
   - Table :ref:`t_mli_sub_tensor_cfg_desc`




   