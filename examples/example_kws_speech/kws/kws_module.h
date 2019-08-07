#ifndef _KWS_MODULE_H_
#define _KWS_MODULE_H_

#include <stdint.h>

#include "kws_types.h"

//
// Base interface of KWS module
//

namespace mli_kws {

//==============================================================
// KWS Module base class.
//
// Usage example:
//      kws_factory builder = ...; // get factory for required module (See kws_factory.h)
//      kws_module *kws;
//      ... // Allocate fast memory and create module by factory
//      ... // open input samples stream
//      do {
//          in_stream.read_samples(&in_samples, kws_cfg.input_samples_num);
//          kws_status status = kws.process(&in_samples, (void *)&fast_mem_a, (void *)&fast_mem_b);
//          if (status == KWS_STATUS_RESULT_READY) {
//              kws_result = kws.get_result();
//              ... // Analyze result
//          } else if (status == KWS_STATUS_ERROR) {
//              ... // Handle error
//              kws.reset()
//          }
//      } while (!in_stream);
//==============================================================
class kws_module {
 public:

    // Virtual destructor to prevent memory leakage
    virtual ~kws_module() = default;

    // Reset module state. Further stream processing will not consider previously passed data
    virtual kws_status reset() = 0;

    // Process current frame of input audio stream
    //
    // params:
    // in_samples      - pointer to input samples. Number of sample must be the kws_info::input_samples_num
    // temp_fast_mem_a - pointer to fast memory A to keep intermediate result.
    //                   Size of memory must be the kws_info::temp_fastmem_a_sz. For better performance mem A
    //                   should be allocated in the different bank to memory B
    // temp_fast_mem_B - pointer to fast memory B to keep intermediate result.
    //                   Size of memory must be the kws_info::temp_fastmem_b_sz. For better performance mem B
    //                   should be allocated in the different bank to memory A
    //
    // return: Status of processing
    // KWS_STATUS_NONE          -  Frame was processed successfully. More samples are required for result
    // KWS_STATUS_RESULT_READY  -  Frame was processed successfully. Result is ready and can be obtainet by get_result()
    // KWS_STATUS_ERROR         -  Error during process.
    virtual kws_status process(const sample_t* in_samples, void* temp_fast_mem_a, void* temp_fast_mem_b) = 0;

    // Get information on key word for stream
    //
    // return: kws_result structure with info on keyword: timestamps boundaries + probability vector
    virtual kws_result get_result() = 0;

    // Return name (c string) of class for particular identefier
    //
    // param: id - numerical identifier of class
    // return: pointer to constant c string with name for passed identifier
    virtual const char* label_id_to_str(int id) const = 0;
};


} // namespace mli_kws {


#endif // #ifndef _KWS_MODULE_H_
