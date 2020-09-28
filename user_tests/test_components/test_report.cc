/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include "test_report.h"

#include <cstdio>
#include <assert.h>


namespace mli {
namespace tst {

// Separator string and it's length. All fields sizes assume it is fixed. 
//==========================================================
static const int kSeparatorStringLength = 111;
static const char kSeparatorString[kSeparatorStringLength] =
    "|============================================================================================================|";

//Print header of test report
//==========================================================
void reporter_full::report_header(const char* case_descr) const {  
    assert(kSeparatorStringLength == 111);
    printf("\n%s\n", kSeparatorString);
    printf("|  %-106s|\n", case_descr);
    printf("%s\n", kSeparatorString);
    printf("| %-30s| %-8s| %-9s|  %-7s| %-10s | %-10s| %-19s|\n",
           "Test Case", "Result", "|S|/|N| ", "SNR[dB]", "|MaxErr|", /*"|N|/|Qer|", */"Qerr[%]", "CRC32 (Status)");
    printf("%s\n", kSeparatorString);
}

// evaluate provided results and populate report table field accrdingly
//==========================================================
bool reporter_full::evaluate_and_report_case(const char* case_descr, 
                                             const quality_metrics& result, const quality_metrics& threshold,
                                             const crc32_calc& crc_result, const crc32_calc& crc_checksum) const {
    assert(kSeparatorStringLength == 111);
    constexpr int kMetricsToPrint = 4;
    const char* kMetricsPrintfFormat[kMetricsToPrint] = {
        "|%-2s%-8.1f",
        "|%-2s%-7.1f",
        "|%-2s%-10f",
        "|%-2s%-9.2f"
    };
    constexpr quality_metrics::metric_id metrics[kMetricsToPrint] = {
        quality_metrics::kMetricSignalToNoiseRatio,
        quality_metrics::kMetricSignalToNoiseRatioDb,
        quality_metrics::kMetricMaxAbsErr,
        quality_metrics::kMetricQuantErrorPercent
    };
    bool metric_passed[kMetricsToPrint] = { true };

    printf("| %-30s", case_descr);
    
    bool is_case_passed = true;
    bool is_crc_ok = true;
    if (crc_checksum.is_valid()) {
        is_crc_ok = (crc_result.get() == crc_checksum.get());
        is_case_passed = is_case_passed && is_crc_ok;
    }
    for (int i = 0; i < kMetricsToPrint; ++i) {
        metric_passed[i] = result.is_threshold_met(metrics[i], threshold);
        is_case_passed = is_case_passed && metric_passed[i];
    }

    printf("| %-8s", (is_case_passed) ? "PASSED" : "FAILED");

    for (int i = 0; i < kMetricsToPrint; ++i) {
        float cur_metric = result.get_metric_float(metrics[i]);
        if (cur_metric > 99999.f) cur_metric = 99999.f;
        printf(kMetricsPrintfFormat[i], (metric_passed[i]) ? "  " : "!!", cur_metric);
    }
    
    if (crc_checksum.is_valid()) {
        printf("|  0x%.8X %-7s|\n", crc_result.get(), is_crc_ok ? "(OK)" : "(DIFF)");
    } else {
        printf("|  0x%-16.8X|\n", crc_result.get());
    }
    
    return is_case_passed;
}

// print an external message regarding testcase
//==========================================================
void reporter_full::report_message(const char* case_descr, const char* message) const {
    assert(kSeparatorStringLength == 111);
    printf("| %-30s| %-75s|\n", case_descr, message);
}

// print an outline of test repor with external final status
//===========================================================
void reporter_full::report_outline(const char* outline_marker, bool is_passed) const {
    assert(kSeparatorStringLength == 111);
    printf("%s\n", kSeparatorString);
    printf("|======= %-30s: Summary Status: %-8s\n", outline_marker, (is_passed)? "PASSED" : "FAILED");
    printf("%s\n\n", kSeparatorString);
}

} // namespace tst
} // namespace mli

