Advanced Image Processing Group
-------------------------------

The Advanced Image Processing group includes kernels which are used 
in NN architectures for complex image or video processing, doing some 
not trainable pre/post processing of data. These operations might be 
specific for a class or family of NN architectures.

 - :ref:`non_max_supress_prot`

 - :ref:`resize_nn_prot`
 
 - :ref:`resize_bilinear_prot`
 
 - :ref:`proposal_prot`
 
 - :ref:`detection_op_prot`
 
.. _non_max_supress_prot:
 
Non-Maximum Suppression Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces list of bounding boxes and corresponding scores to a given maximum 
using “Fast-NMS” algorithm. Initial list could be originally sorted in 
descending order or sorted internally by kernel.

.. code:: c

   k = 0 
   while num < top_k:
	   b0 = B[k] # get next box
	   B1 = ()
	   for b1 in B[k+1:end]:
		   if IoU(b0, b1) > iou_thd:
			   B1 += b1 # add to overlapped set
       B -= B1 # remove from initial set
	   k++
..
   
Kernels which implement general Non-Maximum Suppression functions have the 
following prototype: 

.. code::

   mli_status mli_krn_non_max_supression_<data_format>(
      const mli_tensor *in_boxes,
      const mli_tensor *in_scores,
      const mli_non_max_supression_cfg *cfg,	
      mli_tensor *out);	
..

where ``data_format`` is one of the data formats listed in table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +---------------+-----------------------------------+-----------------------------------------------------------------+
   | **Parameter** | **Type**                          | **Description**                                                 |
   +===============+===================================+=================================================================+
   | ``in_boxes``  | ``mli_tensor *``                  | [IN] Pointer to constant input tensor with boxes coordinates.   |
   +---------------+-----------------------------------+-----------------------------------------------------------------+
   | ``in_scores`` | ``mli_tensor *``                  | [IN] Pointer to constant input tensor with scores per each box. |
   +---------------+-----------------------------------+-----------------------------------------------------------------+
   | ``cfg``       | ``mli_non_max_supression_cfg *``  | [IN] Pointer to parameters structure.                           |
   +---------------+-----------------------------------+-----------------------------------------------------------------+
   | ``out``       | ``mli_tensor*``                   | [OUT] Pointer to output tensor. Result is stored here.          |
   +---------------+-----------------------------------+-----------------------------------------------------------------+
..

``mli_non_max_supression_cfg`` is defined as:

.. code::

   typedef struct {
        mli_data_container scratch_data;
        uint32_t pre_nms_top_k;
        uint32_t post_nms_top_k;
        mli_data_container confidence_threshold;
        bool? input_is_sorted;
        bool? normalized_coord;
        // bool? boxes_coord_type; //(corner vs size&coord);
   } mli_non_max_supression_cfg;
..

.. _t_mli_non_max_supression_cfg_desc:
.. table:: mli_non_max_supression_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | **Field name**           | **Type**               | **Description**                                                         |
   +==========================+========================+=========================================================================+
   |                          |                        | Container with a scratch memory to keep kernel intermediate             |
   | ``scratch_data``         | ``mli_data_container`` | results. Must contain a valid pointer in pi32 field to a memory         |
   |                          |                        | of sufficient   size for kernel (is defined in a kernel description).   |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | ``confidence_threshold`` | ``mli_data_container`` | Container with a single value of the same type as in_scores tensor.     |
   |                          |                        | Shares it’s element type parameters and could be interpreted using it.  |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | ``post_nms_topk``        | ``uint32_t``           |                                                                         |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | ``pre_nms_topk``         | ``uint32_t``           |                                                                         |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | ``input_is_sorted``      | ``bool``               |                                                                         |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
   | ``normalized_coord``     | ``bool``               |                                                                         |
   +--------------------------+------------------------+-------------------------------------------------------------------------+
..

``in_boxes`` tensor is two-dimensional tensors of shape (num_boxes, 4) and contains the set of 
corner coordinates for boxes in an image. This coordinate are expected to be passed in the 
(y1, x1, y2, x2) order, where y1/x1 and y2/ x2 are the coordinates of any diagonal pair of box 
corners. Number of boxes in tensor is specified by first dimension. 

``in_scores`` tensor is one dimensional tensors of shape (num_boxes) and contains  score or 
confidence per each box in in_boxes tensor.


.. table:: Available Non-Maximum Suppression Functions
   :align: center
   :widths: auto 
   
   +-------------------------------------+-----------------------------------+
   | **Function Name**                   | **Details**                       |
   +=====================================+===================================+
   | ``mli_krn_non_max_supression_sa8``  | All tensors data format: **sa8**  |
   +-------------------------------------+-----------------------------------+
   | ``mli_krn_non_max_supression_fx16`` | All tensors data format: **fx16** |
   +-------------------------------------+-----------------------------------+
..


All these functions must comply to the following conditions:

 - ``in_boxes`` and ``in_scores`` tensors must be valid.
 
Depending on the debug level (see section :ref:`err_codes`) this function performs a parameter 
check and return the result as an ``mli_status`` code as described in section :ref:`kernl_sp_conf`.

.. _resize_nn_prot:

Resize Nearest Neighbor Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernels which implement Resize Nearest Neighbor functions have the following prototype:

.. code::

   mli_status mli_krn_resize_nn_<data_format>(
      const mli_tensor *in,
      const mli_non_max_supression_cfg *cfg,	
      mli_tensor *out);	
..

where ``data_format`` is one of the data formats listed in table :ref:`mli_data_fmts` and the function parameters 
are shown in the following table:

.. table:: Data Format Naming Convention Fields
   :align: center
   :widths: auto 
   
   +---------------+-----------------------------------+----------------------------------------+
   | **Parameter** | **Type**                          | **Description**                        |
   +===============+===================================+========================================+
   | ``in``        | ``mli_tensor *``                  | [IN] Pointer to constant input tensor. |
   +---------------+-----------------------------------+----------------------------------------+
   | ``cfg``       | ``mli_resize_bilinear_nn_cfg *``  | [IN] Pointer to parameters structure.  |
   +---------------+-----------------------------------+----------------------------------------+
   | ``out``       | ``mli_tensor *``                  | [OUT] Pointer to output tensor.        |
   |               |                                   | Result is stored here.                 |
   +---------------+-----------------------------------+----------------------------------------+
..

``mli_resize_bilinear_nn_cfg`` is defined as:

.. code::

   typedef struct {
        bool? enum_type? align_corners;
   } mli_resize_bilinear_nn_cfg;
..

.. _t_mli_resize_bilinear_nn_cfg_desc:
.. table:: mli_resize_bilinear_nn_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +-------------------+--------------------+------------------------------------------+
   | **Field name**    | **Type**           | **Description**                          |
   +===================+====================+==========================================+
   | ``align_corners`` | ``Bool?enum_type`` | To align perception field by the center  |
   |                   |                    | of pixel or corner.                      |
   +-------------------+--------------------+------------------------------------------+
..

.. table:: Available Resize Bilinear Nearest Neighbor Functions
   :align: center
   :widths: auto 
   
   +----------------------------+-----------------------------------+
   | **Function Name**          | **Details**                       |
   +============================+===================================+
   | ``mli_krn_resize_nn_sa8``  | All tensors data format: **sa8**  |
   +----------------------------+-----------------------------------+
   | ``mli_krn_resize_nn_fx8``  | All tensors data format: **fx8**  |
   +----------------------------+-----------------------------------+
   | ``mli_krn_resize_nn_fx16`` | All tensors data format: **fx16** |
   +----------------------------+-----------------------------------+
..

All the listed functions must comply to the following conditions:

 - ``in`` tensor must be valid.

.. _resize_bilinear_prot:

Resize Bilinear Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernels which implement Resize-Bilinear functions have the following prototype:

.. code::

   mli_status mli_krn_resize_bilinear_<data_format>(
      const mli_tensor *in,
      const mli_non_max_supression_cfg *cfg,	
      mli_tensor *out);	
   
where ``data_format`` is one of the data formats listed in table :ref:`mli_data_fmts` and the function parameters
are shown in the following table:

.. table:: Resize Bilinear Functions Parameter Description
   :align: center
   :widths: auto 
   
   +---------------+-----------------------------------+-----------------------------------------+
   | **Parameter** | **Type**                          | **Description**                         |
   +===============+===================================+=========================================+
   | ``in``        | ``mli_tensor *``                  | [IN] Pointer to constant input tensor.  |
   +---------------+-----------------------------------+-----------------------------------------+
   | ``cfg``       | ``mli_resize_bilinear_nn_cfg *``  | [IN] Pointer to parameters structure.   |
   +---------------+-----------------------------------+-----------------------------------------+
   | ``out``       | ``mli_tensor *``                  | [OUT] Pointer to output tensor.         |
   |               |                                   | Result is stored here.                  |
   +---------------+-----------------------------------+-----------------------------------------+ 
..

See :ref:`t_mli_resize_bilinear_nn_cfg_desc` for more details on configuration structure.

.. table:: List of Available Resize Bilinear Functions
   :align: center
   :widths: auto 
   
   +----------------------------------+-----------------------------------+
   | **Function Name**                | **Details**                       |
   +==================================+===================================+
   | ``mli_krn_resize_bilinear_sa8``  | All tensors data format: **sa8**  |
   +----------------------------------+-----------------------------------+
   | ``mli_krn_resize_bilinear_fx8``  | All tensors data format: **fx8**  |
   +----------------------------------+-----------------------------------+
   | ``mli_krn_resize_bilinear_fx16`` | All tensors data format: **fx16** |
   +----------------------------------+-----------------------------------+ 
..

All Resize Bilinear functions must comply to the following conditions:

 - ``in`` tensor must be valid.

.. _proposal_prot:

Proposal Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernels which implement RPN Proposal functions have the following prototype:

.. code::

   mli_status mli_krn_proposal_<data_format>(
      const mli_tensor *rpn_scores,
      const mli_tensor *rpn_deltas,
      const mli_tensor *anchor_scale,
      const mli_tensor *anchor_ratio,
      const mli_rpn_cfg *cfg,	
      mli_tensor *out);
..
   
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. table:: RPN Proposal Functions Parameter Description
   :align: center
   :widths: auto 
   
   +------------------+--------------------+--------------------------------------------------------------+
   | **Parameter**    | **Type**           | **Description**                                              |
   +==================+====================+==============================================================+
   | ``rpn_scores``   | ``mli_tensor *``   | [IN] Pointer to constant tensor with region proposal scores. |
   +------------------+--------------------+--------------------------------------------------------------+
   | ``rpn_deltas``   | ``mli_tensor *``   | [IN] Pointer to constant tensor with region proposal deltas. |
   +------------------+--------------------+--------------------------------------------------------------+
   | ``anchor_scale`` | ``mli_tensor *``   | [IN] Pointer to constant tensor with ahchor scales.          |
   +------------------+--------------------+--------------------------------------------------------------+
   | ``anchor_ratio`` | ``mli_tensor *``   | [IN] Pointer to constant tensor with anchor ratios.          |
   +------------------+--------------------+--------------------------------------------------------------+
   | ``cfg``          | ``mli_rpn_cfg *``  | [IN] Pointer to parameters structure.                        |
   +------------------+--------------------+--------------------------------------------------------------+
   | ``out``          | ``mli_tensor *``   | [OUT] Pointer to output tensor. Result is stored here.       |
   +------------------+--------------------+--------------------------------------------------------------+
..

``mli_proposal_cfg`` is defined as:

.. code::

   typedef struct {
        mli_data_container scratch_data;
        uint32_t feat_stride;
        uint32_t ancor_base_size;
        uint32_t max_rois;
   } mli_proposal_cfg;
..

.. _t_mli_proposal_cfg_desc:
.. table:: mli_proposal_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +---------------------+------------------------+-----------------+
   | **Field name**      | **Type**               | **Description** |
   +=====================+========================+=================+
   | ``scratch_data``    | ``mli_data_container`` |                 |
   +---------------------+------------------------+-----------------+
   | ``feat_stride``     | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+
   | ``ancor_base_size`` | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+
   | ``max_rois``        | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+ 
..

.. table:: List of Available Proposal Functions
   :align: center
   :widths: auto 
   
   +---------------------------+-----------------------------------+
   | **Function Name**         | **Details**                       |
   +===========================+===================================+
   | ``mli_krn_proposal_sa8``  | All tensors data format: **sa8**  |
   +---------------------------+-----------------------------------+
   | ``mli_krn_proposal_fx8``  | All tensors data format: **fx8**  |
   +---------------------------+-----------------------------------+
   | ``mli_krn_proposal_fx16`` | All tensors data format: **fx16** |
   +---------------------------+-----------------------------------+   
..

.. _detection_op_prot:

Detection Output Prototype and Function List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernels which implement DetectionOutput functions have the following 
prototype:

.. code::

   mli_status mli_krn_detection_output_<data_format>(
      const mli_tensor **inputs,
   //   const mli_tensor *rpn_deltas,
   //   const mli_tensor *anchor_scale,
   //   const mli_tensor *anchor_ratio,
      const mli_detection_output_cfg *cfg,	
      mli_tensor *out);	
..
   
where ``data_format`` is one of the data formats listed in Table :ref:`mli_data_fmts` and the 
function parameters are shown in the following table:

.. table:: Detection Output Functions Parameter Description
   :align: center
   :widths: auto 
   
   +---------------+---------------------------------+----------------------------------------------+
   | **Parameter** | **Type**                        | **Description**                              |
   +===============+=================================+==============================================+
   | ``inputs``    | ``mli_tensor **``               | [IN] Pointer to constant tensor with region  |
   |               |                                 | proposal scores.                             |
   +---------------+---------------------------------+----------------------------------------------+
   | ``cfg``       | ``mli_detection_output_cfg *``  | [IN] Pointer to parameters structure.        |
   +---------------+---------------------------------+----------------------------------------------+
   | ``out``       | ``mli_tensor *``                | [OUT] Pointer to output tensor. Result is    |
   |               |                                 | stored here.                                 |
   +---------------+---------------------------------+----------------------------------------------+
..

.. code::

   mli_detection_output_cfg is defined as:
   typedef struct {
        mli_data_container scratch_data;
        uint32_t num_inputs;
        uint32_t num_classes;
        uint32_t background_label_id;
        uint32_t keep_top_k;
        uint32_t top_k;
        float? confidence_threshold;
        enum_type code_type;
        float? nms_threshold; -?
        bool share_location;
        bool variance_encoded_in_target; 
   } mli_proposal_cfg;
..

.. _t_mli_detection_output_cfg_desc:
.. table:: mli_detection_output_cfg Structure Field Description
   :align: center
   :widths: auto 
   
   +---------------------+------------------------+-----------------+
   | **Field name**      | **Type**               | **Description** |
   +=====================+========================+=================+
   | ``scratch_data``    | ``mli_data_container`` |                 |
   +---------------------+------------------------+-----------------+
   | ``feat_stride``     | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+
   | ``ancor_base_size`` | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+
   | ``max_rois``        | ``uint32_t``           |                 |
   +---------------------+------------------------+-----------------+
..

.. table:: List of Available Proposal Functions
   :align: center
   :widths: auto 
   
   +---------------------------+-----------------------------------+
   | **Function Name**         | **Details**                       |
   +===========================+===================================+
   | ``mli_krn_proposal_sa8``  | All tensors data format: **sa8**  |
   +---------------------------+-----------------------------------+
   | ``mli_krn_proposal_fx8``  | All tensors data format: **fx8**  |
   +---------------------------+-----------------------------------+
   | ``mli_krn_proposal_fx16`` | All tensors data format: **fx16** |
   +---------------------------+-----------------------------------+
..


