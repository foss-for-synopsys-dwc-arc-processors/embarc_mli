MLI Profiling Mechanism
=======================

Introduction
------------

This chapter describes the mechanisms used to profile the performance 
of the MLI software stack.

The purposes of profiling are threefold. For each purpose, MLI is profiled 
at a different level/granularity:

 - Customer engagement – profiled at graph level for comparing end-to-end 
   solutions.

 - Regression testing – profiled at layer level for ensuring traceability 
   in performance improvements/degradations

 - Performance debugging – profiled at the level of fine-grain events such 
   as DMA transfers, compute blocks, and so on
   
In addition to profiling, efficient compilation and scheduling of graphs 
requires estimates of costs of running kernels in terms of different metrics. 
This is discussed in (TODO: add ref)

Profiling infrastructure
------------------------

All profiling MLI functions start with a ``mli_prof_ prefix``.

mli_prof_get_metric()
~~~~~~~~~~~~~~~~~~~~~

Measurements are obtained by calling ``mli_prof_get_metric()``.

MLI implementations are responsible for providing a platform-specific implementation of this function.

.. code::

   void mli_prof_get_metric(mli_prof_metric metric, mli_prof_measurement* measurement);
..

.. table:: mli_prof_get_metric Parameters
   :align: center
   :widths: auto
   
   +--------------------------+----------------------------------------------------------------+
   | **Parameter Name**       | **Description**                                                |
   +==========================+================================================================+
   | ``mli_prof_metric``      | Enum value identifying the metric to be measured (see the      |
   |                          | following code)                                                |
   +--------------------------+----------------------------------------------------------------+
   | ``mli_prof_measurement`` | Pointer to the preallocated space for the measurement result.  |
   |                          | The allocated space fits a single instance of the              |
   |                          | ``mli_prof_measurement`` struct.                               |
   +--------------------------+----------------------------------------------------------------+
..

The ``mli_prof_metric`` enum is defined as follows:

.. code::

   typedef enum {
     MLI_PROF_METRIC_CYCLES,
     MLI_PROF_METRIC_DATA_INBOUND,
     MLI_PROF_METRIC_DATA_OUTBOUND

   } mli_prof_metric;
..

The enum values are used to identify the following metrics respectively:

 - Cycle count on the given core

 - Total amount of data transferred from external to local memory

 - Total amount of data transferred from local to external memory

New metrics might be added to the enum in this document. MLI does not support 
measuring arbitrary metrics outside of this specification.

The measured numbers are stored in mli_prof_measurement data structures:

.. code::

   typedef struct __mli_prof_measurement {
     mli_prof_time timestamp;
     uint64_t measurement;  // or whichever the widest unsigned integer
   } mli_prof_measurement;

The mli_prof_identifier struct contains the kernel and metric ID that 
pertain to the measurement: 

.. code::

   typedef struct __mli_prof_identifier {
     mli_prof_kernel_id kernel;
     mli_prof_metric metric;
   } mli_prof_identifier;
..

The measurement struct is bundled with mli_prof_identifier struct into a 
``mli_prof_datapoint`` container, whose preparation is the task of the 
MLI API caller:

.. code::

   typedef struct __mli_prof_datapoint {
     mli_prof_identifier identifier;
     mli_prof_measurement measurement;
   } mli_prof_datapoint;
..

mli_prof_get_time()
~~~~~~~~~~~~~~~~~~~

To provide an architecture-specific time measurement functionality, 
the MLI implementation includes ``mli_prof_get_time()`` with the 
following prototype:

.. code::

   void mli_prof_get_time(mli_prof_time* time);
..

.. table:: mli_prof_get_time Parameters
   :align: center
   :widths: auto
   
   +--------------------+----------------------------------------------------+
   | **Parameter Name** | **Description**                                    |
   +====================+====================================================+
   | **time**           | Pointer to a structure to write the timestamp into |
   +--------------------+----------------------------------------------------+
..

``mli_prof_time`` is a wrapper around the system timestamp holding structures.

.. admonition:: Example 
   :class: "admonition tip"
   
   The ``mli_prof_time`` on a Linux system might look similar to this:
   
   .. code::
   
      struct mli_prof_time {
        struct timespec time_data;
      };
   ..  
..

MLI implementations must also provide a function for interpreting time intervals:

.. code::

   uint_64t mli_prof_time_interval_usec(mli_prof_time *start, mli_prof_time *end);
..

.. table:: mli_prof_time_interval_usec Parameters
   :align: center
   :widths: auto
   
   +--------------------+------------------------------------------------+
   | **Parameter Name** | **Description**                                |
   +====================+================================================+
   | **start**          | Pointer to a structure holding the interval’s  |
   |                    | starting timestamp                             |
   +--------------------+------------------------------------------------+
   | **end**            | Pointer to a structure holding the interval’s  |
   |                    | ending timestamp                               |
   +--------------------+------------------------------------------------+
   | **Return value**   | Time elapsed between the two timestamps in     |
   |                    | microseconds.                                  |
   +--------------------+------------------------------------------------+
..

These functions allow a generic API to be used which abstracts away any 
implementation-specific notion of time storage.

The responsibility of measuring performance is on caller of the MLI API.

