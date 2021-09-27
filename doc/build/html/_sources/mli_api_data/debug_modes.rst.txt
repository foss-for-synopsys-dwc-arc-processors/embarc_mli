.. _dbg_modes:

Debug Modes
-----------

The MLI library supports multiple debug levels. At the beginning of each function, the 
input parameters can be checked and the result is passed as a return code. This 
helps to detect user errors. Inside, the library preconditions and invariants are 
checked with asserts. Also, extra messages can be printed to make debugging easier. 
Because all this debug code also affects the performance, it can be enabled at the 
compile time in 5 possible levels by setting the ``MLI_DEBUG_MODE`` preprocessor define 
as follows:

 - ``DBG_MODE_RELEASE`` (``MLI_DEBUG_MODE = 0``) - No debug. Functions do not examine parameters. 
   Data is processed with assumption that function input is valid. This might lead to 
   undefined behavior if the assumption is not true. Functions always return ``MLI_STATUS_OK``. 
   No messages are printed, and no assertions are used.
   
 - ``DBG_MODE_RET_CODES`` (``MLI_DEBUG_MODE = 1``) - Functions examine parameters and return valid 
   error status if any violation of data is found. Otherwise, functions process data and return 
   status ``MLI_STATUS_OK``. No messages are printed and no assertions are used.
   
 - ``DBG_MODE_ASSERT`` (``MLI_DEBUG_MODE = 2``) - Functions examine parameters. Any violations of 
   data lead to a break on assert().
   
 - ``DBG_MODE_DEBUG`` (``MLI_DEBUG_MODE = 3``) - The same as ``DBG_MODE_ASSERT``, but before 
   breaking on ``assert()`` function prints descriptive message using standard ``printf()``.
   
 - ``DBG_MODE_FULL`` (``MLI_DEBUG_MODE = 4``) - The same as ``DBG_MODE_DEBUG``, but additionally 
   extra assertions inside loops are used for this mode.
    
By default, ``MLI_DEBUG_MODE`` is set to ``DBG_MODE_RELEASE``. The following table summarizes modes behavior.

.. table:: MLI_DEBUG_MODE Modes Behavior
   :align: center
   :widths: 30, 30, 30, 30, 30, 30 
   
   +------------------+-------------+---------------+------------+-----------+----------+
   | **Behaviour**    | **RELEASE** | **RET_CODES** | **ASSERT** | **DEBUG** | **FULL** |
   +==================+=============+===============+============+===========+==========+
   | Return Codes     | NO          | YES           | YES        | YES       | YES      |      
   +------------------+-------------+---------------+------------+-----------+----------+
   | Assertions       | NO          | NO            | YES        | YES       | YES      | 
   +------------------+-------------+---------------+------------+-----------+----------+
   | Messages         | NO          | NO            | NO         | YES       | YES      | 
   +------------------+-------------+---------------+------------+-----------+----------+
   | Extra Assertions | NO          | NO            | NO         | NO        | YES      | 
   +------------------+-------------+---------------+------------+-----------+----------+
..
