#
# Copyright 2020-2022, Synopsys, Inc.
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
    elseif (${MLI_PLATFORM} STREQUAL EM_HS)
        set(ROUND_MODE CONVERGENT)
    elseif (${MLI_PLATFORM} STREQUAL ARC_NODSP_NOVDSP)
        set(ROUND_MODE UP)
    else()
        message(FATAL_ERROR "Please specify a rounding mode: UP or CONVERGENT")
    endif()
endif()

if(ROUND_MODE STREQUAL UP)
    list(APPEND TEST_COMPILE_DEFINITIONS CRC_RM_UP)
elseif(ROUND_MODE STREQUAL CONVERGENT)
    list(APPEND TEST_COMPILE_DEFINITIONS CRC_RM_CONVERGENT)
else()
    message(FATAL_ERROR "Rounding mode ${ROUND_MODE} is not supported")
endif()

if (FULL_ACCU STREQUAL ON)
    list(APPEND TEST_COMPILE_DEFINITIONS FULL_ACCU_ON)
endif()

# Note: AVEPOOL_16BIT_MUL should be same as the one in mli_lib.cmake
if (NOT DEFINED AVEPOOL_16BIT_MUL)
    set(AVEPOOL_16BIT_MUL OFF)
endif()

if(AVEPOOL_16BIT_MUL STREQUAL ON)
    list(APPEND TEST_COMPILE_DEFINITIONS AVEPOOL_16BIT_MUL)
elseif(AVEPOOL_16BIT_MUL STREQUAL OFF)
    # we don't do anything in this case
else()
    message(FATAL_ERROR "Please specify AVEPOOL_16BIT_MUL : ON or OFF")
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
    test_components/test_rescale_utility.cc
    test_components/test_tiling.cc
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
# Function to add new user test
#======================================================
function(ADD_USER_TEST MLI_SET TEST_NAME)
    # Support an optional argument 'DATA_TYPE' to this function.
    if (${ARGC} GREATER 2)
        set(DATA_TYPE ${ARGV2})
        set(TARGET test_mli_${MLI_SET}_${TEST_NAME}_${DATA_TYPE})
    else()
        set(DATA_TYPE)
        set(TARGET test_mli_${MLI_SET}_${TEST_NAME})
    endif()

    add_executable(${TARGET}
        tests/mli_${MLI_SET}_${TEST_NAME}/tests_mli_${MLI_SET}_${TEST_NAME}.cc
    )

    target_include_directories(${TARGET} PRIVATE
        tests/mli_${MLI_SET}_${TEST_NAME}
    )

    target_link_libraries(${TARGET} PUBLIC mli)
    target_link_libraries(${TARGET} PUBLIC test_components_obj)

    target_compile_options(${TARGET} PRIVATE
        ${MLI_PLATFORM_COMPILE_OPTIONS}
        ${TEST_COMPILE_OPTIONS}
        ${DSP_CONTROL_OPTION}
    )
    target_compile_definitions(${TARGET} PRIVATE ${TEST_COMPILE_DEFINITIONS})

    if (NOT ${DATA_TYPE} STREQUAL "")
        target_compile_definitions(${TARGET} PRIVATE
            COMPILE_FOR_${DATA_TYPE})
    endif()

    target_link_options(${TARGET} PRIVATE
        ${MLI_PLATFORM_LINK_OPTIONS}
        ${TEST_COMPILE_OPTIONS}
        ${DSP_CONTROL_OPTION}
    )

    if (ARC)
        target_link_options(${TARGET} PRIVATE
            -m
            -Coutput=./bin/${TARGET}.map
        )
    endif()

    set_target_properties(${TARGET}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
    )
endfunction()

#======================================================
# Helpers functions Group
#======================================================
add_user_test(hlp convert_tensor)
add_user_test(hlp tensor_struct)

#======================================================
# Data Movement Group
#======================================================
add_user_test(krn data_movement)
add_user_test(krn data_movement_30)
add_user_test(krn move_broadcast_30)

#======================================================
# Convolution Group
#======================================================
add_user_test(krn conv2d)
add_user_test(krn conv2d_30)
add_user_test(krn depthwise_conv)
add_user_test(krn depthwise_conv_30)
add_user_test(krn transpose_conv2d)
add_user_test(krn transpose_conv2d_30)
add_user_test(krn group_conv2d FX16)
add_user_test(krn group_conv2d FX16_FX8_FX8)
add_user_test(krn group_conv2d SA8_SA8_SA32)
add_user_test(krn matmul_30)

#======================================================
# Pooling Group
#======================================================
add_user_test(krn avepool)
add_user_test(krn maxpool)
add_user_test(krn maxpool_30)
add_user_test(krn sumpool_30)

#======================================================
# Diverse Kernels Group
#======================================================
add_user_test(krn argmax)
add_user_test(krn permute)
add_user_test(krn permute_30)

#======================================================
# Fully Connected And Recurrent Group
#======================================================
add_user_test(krn fully_connected)
add_user_test(krn fully_connected_30)
add_user_test(krn rnn_dense)
add_user_test(krn lstm_cell FX16)
add_user_test(krn lstm_cell FX16_FX8_FX8)
add_user_test(krn lstm_cell SA8_SA8_SA32)
add_user_test(krn gru_cell FX16)
add_user_test(krn gru_cell FX16_FX8_FX8)
add_user_test(krn gru_cell SA8_SA8_SA32)
#======================================================
# Transform (Activation) Group
#======================================================
add_user_test(krn softmax)
add_user_test(krn relu)
add_user_test(krn leaky_relu)
add_user_test(krn prelu)
add_user_test(krn tanh)
add_user_test(krn sigm)
add_user_test(krn l2_normalize)

#======================================================
# Eltwise Group
#======================================================
add_user_test(krn eltwise)
add_user_test(krn eltwise_30)

#======================================================
# Reduce Group
#======================================================
add_user_test(krn reduce_max_30)
add_user_test(krn reduce_sum_30)

#======================================================
# Resize Group
#======================================================
add_user_test(krn resize_bilinear_30)
