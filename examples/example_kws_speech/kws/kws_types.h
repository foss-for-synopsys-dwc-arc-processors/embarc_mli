#ifndef _KWS_MODULE_TYPES_H_
#define _KWS_MODULE_TYPES_H_

#include <stdint.h>


//
// Public interface types shared between KWS modules
//

#ifdef __cplusplus
extern "C" {
#endif

// Default input sample and timestamp types for all KWS modules
typedef int16_t sample_t;
typedef uint32_t timestamp_t;

//==============================================================
// KWS Module status enumeration.
//
//==============================================================
typedef enum {
    KWS_STATUS_NONE = 0,           // Function finished normally
    KWS_STATUS_RESULT_READY,       // Result vector ready and can be obtained by get_result method
    KWS_STATUS_ERROR,              // Error in processing processing
} kws_status;

//==============================================================
// KWS Module information.
//
// Contains various information for proper module initialization/usage. To be obtainde befor usage.
//==============================================================
typedef struct {
    uint32_t input_samples_num;     // Number of samples to be passed to process method
    uint32_t output_values_num;     // Number of values in output vector to be obtained by get_result method
    uint32_t timestamp_duration_ms; // Time duration between two sequential timestamps in KWS module

    uint32_t state_fastmem_a_sz;    // Size of fast memory to keep KWS module state. To be provided to building function.
                                    // Used during module lifetime (must not be used externally).
    uint32_t temp_fastmem_a_sz;     // Size of fast memory in bank A to store data during single process function.
                                    // To be provided to process method. Used during process function execution.
                                    // Can be used outside processing.
    uint32_t temp_fastmem_b_sz;     // Size of fast memory in bank B to stor data during single process function.
                                    // Usage is similar to fastmem A.

    uint32_t coeff_fastmem_sz;      // Information on used memory for coefficients (section .mli_model)
                                    // to be allocated in compile time.
    uint32_t dynamic_mem_sz;        // Information on size of dynamically  allocated memory by model (initialization only)
    int silence_id;                 // Silence ID
    const char* name;               // Name of module
} kws_info;

//==============================================================
// KWS Module output structure.
//
// Contains information on found key word in stream: time frame and probability vector.
//==============================================================
typedef struct {
    timestamp_t start;
    timestamp_t end;
    const float *results;
} kws_result;


#ifdef __cplusplus
} //extern "C"
#endif


#endif // #ifndef _KWS_MODULE_TYPES_H_
