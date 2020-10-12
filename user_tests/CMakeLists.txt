#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

project(mli_user_tests)

include(../cmake/settings.cmake)

#======================================================
# Set User Tests specific flags and options:
#======================================================
#   TEST_COMPILE_OPTIONS - test specific compilation options
#
# TODO: Extra options regarding assertions inside tests 
# need to be taken into account during reconfiguration

# Note: The ROUND_MODE and DSP_CONTROL_OPTION should be shared on upper level (settings.cmake)
set(TEST_COMPILE_OPTIONS)
set(TEST_COMPILE_DEFINITIONS)
set(DSP_CONTROL_OPTION)
if (NOT DEFINED ROUND_MODE)
    if(${MLI_PLATFORM} STREQUAL VPX)
        set(ROUND_MODE UP)
    else()
        set(ROUND_MODE CONVERGENT)
    endif()
endif()


if(ROUND_MODE STREQUAL UP)
    list(APPEND TEST_COMPILE_DEFINITIONS CRC_RM_UP)
elseif(ROUND_MODE STREQUAL CONVERGENT)
    list(APPEND TEST_COMPILE_DEFINITIONS CRC_RM_CONVERGENT)
else()
    message(FATAL_ERROR "Rounding mode ${ROUND_MODE} is not supported")
endif()

if (ARC)
    list(APPEND TEST_COMPILE_OPTIONS
        -Hnocopyr
        -Hpurge
        -Hheap=16K
        -Hstack=16K
        -e_start
        -Bgrouplib
        -Hldopt=-q
        -Hsdata0
        -Hdense_prologue
    )

    if (${MLI_PLATFORM} STREQUAL VPX)
        list(APPEND DSP_CONTROL_OPTION
            "SHELL: -mllvm -slot_swapping=true")

    elseif (${MLI_PLATFORM} STREQUAL EM_HS)
        if(ROUND_MODE STREQUAL UP)
            list(APPEND DSP_CONTROL_OPTION
                -Xdsp_ctrl=postshift,guard,up
                -Hfxapi
            )
        elseif(ROUND_MODE STREQUAL CONVERGENT)
            list(APPEND DSP_CONTROL_OPTION
                -Xdsp_ctrl=postshift,guard,convergent
                -Hfxapi
            )
        endif()

    else()
        message(WARNING "ARC kind of platform isn't supported - can't provide proper test defines")
    endif()
endif() # ARC

#======================================================
# Test Components as an object library target
#======================================================
add_library(test_components_obj OBJECT
    test_components/test_crc32_calc.cc
    test_components/test_memory_manager.cc
    test_components/test_quality_metrics.cc
    test_components/test_tensor_quantizer.cc
    test_components/test_report.cc
    ../examples/auxiliary/tensor_transform.c
    ../examples/auxiliary/tests_aux.c
    ../examples/auxiliary/idx_file.c
)

target_compile_options(test_components_obj PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${DSP_CONTROL_OPTION}
)

target_compile_definitions(test_components_obj PRIVATE ${TEST_COMPILE_DEFINITIONS})

set(test_components_inc
    test_components
    ../examples/auxiliary
)

get_property(mli_include_dirs TARGET mli PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(test_components_obj PUBLIC ${test_components_inc})
target_include_directories(test_components_obj PRIVATE ${mli_include_dirs})

#======================================================
# Convolution 2D Tests
#======================================================
add_executable(test_mli_krn_conv2d
    tests/mli_krn_conv2d/tests_mli_krn_conv2d.cc
)

target_include_directories(test_mli_krn_conv2d PRIVATE
    tests/mli_krn_conv2d
)

target_link_libraries(test_mli_krn_conv2d PUBLIC mli)
target_link_libraries(test_mli_krn_conv2d PUBLIC test_components_obj)

target_compile_options(test_mli_krn_conv2d PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${DSP_CONTROL_OPTION}
)
target_compile_definitions(test_mli_krn_conv2d PRIVATE ${TEST_COMPILE_DEFINITIONS})

target_link_options(test_mli_krn_conv2d PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${DSP_CONTROL_OPTION}
)

if (ARC)
    target_link_options(test_mli_krn_conv2d PRIVATE
        -m
        -Coutput=./bin/test_mli_krn_conv2d.map
    )
endif()

set_target_properties(test_mli_krn_conv2d
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
)

#======================================================
# Softmax
#======================================================
add_executable(test_mli_krn_softmax
    tests/mli_krn_softmax/tests_mli_krn_softmax.cc
)

target_include_directories(test_mli_krn_softmax PRIVATE
    tests/mli_krn_softmax
)

target_link_libraries(test_mli_krn_softmax PUBLIC mli)
target_link_libraries(test_mli_krn_softmax PUBLIC test_components_obj)

target_compile_options(test_mli_krn_softmax PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${DSP_CONTROL_OPTION}
)

target_compile_definitions(test_mli_krn_softmax PRIVATE ${TEST_COMPILE_DEFINITIONS})

target_link_options(test_mli_krn_softmax PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${DSP_CONTROL_OPTION}
)

if (ARC)
    target_link_options(test_mli_krn_softmax PRIVATE
        -m
        -Coutput=./bin/test_mli_krn_softmax.map
    )
endif()

set_target_properties(test_mli_krn_softmax
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
)

