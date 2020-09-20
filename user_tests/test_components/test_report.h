/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_REPORT_H_
#define _MLI_USER_TESTS_TEST_REPORT_H_

#include "mli_api.h"
#include "test_metrics.h"

namespace mli {
namespace tst {

//===============================================================================================
// Mock reporter
//===============================================================================================
//class reporter {
// public:
//     void report_header(const char* case_descr) {};
//
//     bool evaluate_and_report_case(const char* case_descr, const quality_metrics& result,
//                                   const quality_metrics& threshold, int32_t crc_result) {
//         return true;
//     };
//
//     void report_message(const char* case_descr, const char* message) {};
//
//     void report_outline(bool status) {};
//};

class reporter_full /*: public reporter */{
public:
    void report_header(const char* case_descr) const;

    bool evaluate_and_report_case(const char* case_descr, 
                                  const quality_metrics& result, const quality_metrics& threshold, 
                                  const crc32_calc& crc_result, const crc32_calc& crc_checksum) const;

    void report_message(const char* case_descr, const char* message) const;

    void report_outline(const char* outline_marker, bool is_passed) const;

};


} // namespace mli {
} // namespace tst {

#endif // _MLI_USER_TESTS_TEST_REPORT_H_

