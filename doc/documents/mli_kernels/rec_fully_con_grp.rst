Recurrent and Fully Connected Group
-----------------------------------

This group describes operations related to the RNN calculations and includes 
general fully connected kernel description. There are a lot of variations of 
RNN Cells architectures. MLI provides the most popular ones as a separate kernel. 
RNN Dense kernel in combination with elementwise and transform kernels might be 
used to construct Recurrent cells which are not present in MLI. 

Functions in this group typically use the ``mli_rnn_cell_cfg`` structure, defined as:

.. code::

   typedef struct {
      mli_rnn_direction direction;
      mli_rnn_results results;
      mli_rnn_out_activation act;
      mli_data_container scratch_data;   
      uint32_t scratch_capacity;
    } mli_rnn_cell_cfg;
..

.. _t_mli_rnn_cell_cfg_desc:
.. table:: mli_rnn_cell_cfg Structure Field Description
   :align: center
   :widths: auto 

   +----------------------+------------------------+-----------------------+-----------------------------------------------------------------------+
   | **Field Name**       | **Type**               | **Enumeration Value** | **Description**                                                       |
   +======================+========================+=======================+=======================================================================+
   |                      | ``mli_rnn_direction``  |                       | Process the input sequence in forward direction (from the first       |
   | ``direction``        |                        | ``RNN_DIR_FORWARD``   | entity to the last one).                                              |
   |                      | (enumeration)          |                       |                                                                       |
   |                      |                        +-----------------------+-----------------------------------------------------------------------+
   |                      |                        | ``RNN_DIR_BACKWARD``  | Process the input sequence in backward direction (from the last       |
   |                      |                        |                       | entity to the first one).                                             |
   +----------------------+------------------------+-----------------------+-----------------------------------------------------------------------+
   |                      |                        | ``RNN_OUT_LAST``      | Preserve only the last result after all RNN iterations in output      |
   |                      | ``mli_rnn_results``    |                       | tensor                                                                |
   | ``results``          |                        +-----------------------+-----------------------------------------------------------------------+
   |                      | (enumeration)          |                       |                                                                       |
   |                      |                        | ``RNN_OUT_ALL``       | Preserve result of each RNN iteration in output tensor                |
   +----------------------+------------------------+-----------------------+-----------------------------------------------------------------------+
   |                      |                        |                       | Container with a scratch memory to keep cell’s intermediate results.  |
   | ``scratch_data``     | ``mli_data_container`` | --                    | Must contain a valid pointer in pi32 field to a memory of sufficient  |
   |                      |                        |                       | size for the kernel. The exact amount of memory is defined in the     |
   |                      |                        |                       | respective kernel descriptions in subsequent sections.                |
   +----------------------+------------------------+-----------------------+-----------------------------------------------------------------------+
   | ``scratch_capacity`` | ``uint32_t``           | --                    | Size of a memory pointed by scratch_data field (in bytes).            |
   +----------------------+------------------------+-----------------------+-----------------------------------------------------------------------+
..

.. toctree::
   :maxdepth: 2
   
   rec_fully_con.rst
   rec_rnn_dense.rst
   rec_lstm.rst
   rec_gru.rst




