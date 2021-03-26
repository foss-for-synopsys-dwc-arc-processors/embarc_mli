/*
* Copyright 2019-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _MLI_USER_TESTS_TEST_REPORT_H_
#define _MLI_USER_TESTS_TEST_REPORT_H_

#include "mli_api.h"
#include "test_crc32_calc.h"
#include "test_quality_metrics.h"

namespace mli {
namespace tst {


//===============================================================================================
// Basic test reporter with external message and status:
//
// |============================================================================================================|
// | MLI | Kernels | Convolution 2D  Tests |
// |============================================================================================================|
// | Test Case                     | Result  | Message                                                          |
// |============================================================================================================|
// | Test 1 FX16                   | PASSED  |  Wow! It works                                                   |
// | Test 1 FX16_FX8_FX8           | PASSED  |  This one is also fine. Awesome!                                 |
// | .....................................................................................................      |
// |============================================================================================================|
// |=======[AUTO] Group: mli_krn_conv2d: Summary Status : PASSED
// |============================================================================================================|
//
//  where: 
//      'Test Case' Field - description provided by user
//      'Result' Field - Test case result. PASSED or FAILED depending on bool status provided by user
//      'Message' Field - Message with extra info provided by user
//===============================================================================================
class reporter_basic {
public:
    // Print header of test report in the following way
    // |=============================================================|
    // | <case_descr>                 
    // |=============================================================|
    // | <Table Header - see above>
    // |=============================================================|
    //
    // params:
    // [IN] case_descr - Case description string that will be printed in header (<106 characters).
    //
    // No return;
    void report_header(const char* case_descr) const;

    // Print the case status and additional message in the following way:
    // | <case_descr>       | <status>    |  <message>                            |
    //
    // params:
    // [IN] case_descr - Case description string that will be printed in first field of table (<30 characters)
    // [IN] message - Case description string that will be printed in first field of table (<65 characters)
    // [IN] is_passed - status of the test case (true if passed)
    // 
    // No return
    void report_case(const char* case_descr, const char* message, bool is_passed) const;

    // Print an outline of test repor with external marker string and final status in the following way:
    //|======= <outline_marker>: Summary Status : <is_passed>
    //
    // params:
    // [IN] outline_marker - Case description string that will be printed in first field of table (<30 characters)
    // [IN] is_passed - bool final status (true if passed)
    // 
    // No return
    void report_outline(const char* outline_marker, bool is_passed) const;
};


//===============================================================================================
// Full test reporter with all fields to validate. Example output:
//
// |============================================================================================================|
// | MLI | Kernels | Convolution 2D  Tests |
// |============================================================================================================|
// | Test Case                     | Result  | |S|/|N|  |  SNR[dB]| |MaxErr|   | Qerr[%]   | CRC32 (Status)     |
// |============================================================================================================|
// | Test 1 FX16                   | PASSED  |  10575.5 |  80.5   |  0.000758  |  57.03    |  0x3669E8DA (OK)   |
// | Test 1 FX16_FX8_FX8           | PASSED  |  58.5    |  35.3   |  0.136087  |  0.32     |  0x627FD168 (OK)   |
// | .....................................................................................................      |
// |============================================================================================================|
// |=======[AUTO] Group: mli_krn_conv2d: Summary Status : PASSED
// |============================================================================================================|
//
//  where: 
//      'Test Case' Field - description provided by user
//      'Result' Field - Test case result. PASSED or FAILED depending on other fields and thresholds for them
//      '|S|/|N|' Field - Signal-to-Noise absolute ratio (see quality_metrics class)
//      'SNR[dB]' Field - Signal-to-Noise ratio in decibels (see quality_metrics class)
//      'MaxErr' Field - Maximum error per point (see quality_metrics class)
//      'Qerr[%]' Field  - Percentage of quantization error in total noise (see quality_metrics class)
//      'CRC32 (Status)' Field - CRC32 sum provided by user and it's status in comparison with reference 
//===============================================================================================
class reporter_full {
public:
    // Print header of test report in the following way
    // |=============================================================|
    // | <case_descr>                 
    // |=============================================================|
    // | <Table Header - see above>
    // |=============================================================|
    //
    // params:
    // [IN] case_descr - Case description string that will be printed in header (<106 characters).
    //
    // No return;
    void report_header(const char* case_descr) const;

    // Evaluate provided results and populate report table field accordingly
    //
    // params:
    // [IN] case_descr - Case description string that will be printed in first field of table (<30 characters)
    // [IN] result - quality_metrics instance with measurements calculated on predicted data
    // [IN] threshold - quality_metrics instance with numbers (threshold) for comparison with 'results' instance
    // [IN] crc_result - crc32_calc instance with calculated CRC32 sum calculated on some predicted data
    // [IN] crc_checksum - crc32_calc instance with expected CRC32 sum. Might be not valid.
    // 
    // return true if all metrics from 'result' below threshold from 'threshold' and crc_result equal 
    //        to valid crc_checksum. Return false otherwise.
    bool evaluate_and_report_case(const char* case_descr, 
                                  const quality_metrics& result, const quality_metrics& threshold, 
                                  const crc32_calc& crc_result, const crc32_calc& crc_checksum) const;

    // Print an external message regarding testcase in the following way:
    // | <case_descr>                   |  <message>                            |
    //
    // params:
    // [IN] case_descr - Case description string that will be printed in first field of table (<30 characters)
    // [IN] message - Case description string that will be printed in first field of table (<75 characters)
    // 
    // No return
    void report_message(const char* case_descr, const char* message) const;

    // Print an outline of test repor with external marker string and final status in the following way:
    //|======= <outline_marker>: Summary Status : <is_passed>
    //
    // params:
    // [IN] outline_marker - Case description string that will be printed in first field of table (<30 characters)
    // [IN] is_passed - bool final status (true if passed)
    // 
    // No return
    void report_outline(const char* outline_marker, bool is_passed) const;
};


} // namespace tst
} // namespace mli

#endif // _MLI_USER_TESTS_TEST_REPORT_H_

