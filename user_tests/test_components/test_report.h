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
#include "test_infra.h"

namespace mli {
namespace tst {

//===============================================================================================
// Full test reporter with all fields to validate
//===============================================================================================
class reporter_full {
public:
    // Print header of test report
    void report_header(const char* case_descr) const;

    // Evaluate provided results and populate report table field accrdingly
    bool evaluate_and_report_case(const char* case_descr, 
                                  const quality_metrics& result, const quality_metrics& threshold, 
                                  const crc32_calc& crc_result, const crc32_calc& crc_checksum) const;

    // print an external message regarding testcase
    void report_message(const char* case_descr, const char* message) const;

    // print an outline of test repor with external final status
    void report_outline(const char* outline_marker, bool is_passed) const;
};


} // namespace tst
} // namespace mli

#endif // _MLI_USER_TESTS_TEST_REPORT_H_

