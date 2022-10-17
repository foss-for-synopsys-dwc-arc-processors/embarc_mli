#
# Copyright 2020-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# FLAGS here are similar to lib\make\makefile

if (_MLI_LIB_CMAKE_LOADED)
  return()
endif()
set(_MLI_LIB_CMAKE_LOADED TRUE)

function(_get_cmake_file_dir VAR)
    set(${VAR} ${CMAKE_CURRENT_FUNCTION_LIST_DIR} PARENT_SCOPE)
endfunction()
_get_cmake_file_dir(MLI_LIB_CMAKE_DIR)

include(${MLI_LIB_CMAKE_DIR}/../cmake/settings.cmake)

# To keep code similar to our make files, we use file(GLOB...) to add source files, consider to explicitly add them.
file(GLOB temp
    ${MLI_LIB_CMAKE_DIR}/src/helpers/src/*.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/*.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*hwc*.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/bricks/*.cc
    ${MLI_LIB_CMAKE_DIR}/src/private/src/mli_check.cc
    ${MLI_LIB_CMAKE_DIR}/src/private/src/mli_prv_activation_lut.cc
    ${MLI_LIB_CMAKE_DIR}/src/move/*.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse/*.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*hwc*.cc
)
if (NOT DEFINED MLI_INCLUDE_RUNTIME)
    set(MLI_INCLUDE_RUNTIME ON)
endif()

if ( ${MLI_INCLUDE_RUNTIME} STREQUAL ON)
file(GLOB temp_runtime
    ${MLI_LIB_CMAKE_DIR}/src/private/src/mli_runtime.cc
)
endif()

set(MLI_LIB_SOURCE_FILES
    ${temp}
    ${temp_runtime}
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_relu_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_leaky_relu_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_prelu.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_sigm_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_tanh_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_softmax_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_l2_normalize.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_conv2d_hwcn.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_conv2d_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_conv2d_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_transpose_conv2d_hwcn.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_transpose_conv_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_transpose_conv_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_depthwise_conv2d_hwcn.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_group_conv2d_hwcn.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_depthwise_conv2d_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_depthwise_conv2d_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_matmul_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_matmul_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/mli_krn_fully_connected.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/impl/mli_krn_fully_connected_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/impl/mli_krn_fully_connected_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/mli_krn_rnn_dense.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse/mli_krn_argmax.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse/mli_krn_permute_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/mli_krn_lstm_cell.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/common/mli_krn_gru_cell.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_krn_rescale_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_krn_rescale_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_reduce_max_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_reduce_max_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_reduce_sum_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion/mli_reduce_sum_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/clip/mli_krn_clip_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/clip/mli_krn_clip_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_prelu_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_prelu_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse/mli_krn_permute_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse/mli_krn_permute_runtime.cc
    ${MLI_LIB_CMAKE_DIR}/src/move/mli_move_broadcast_compiler.cc
    ${MLI_LIB_CMAKE_DIR}/src/move/mli_move_broadcast_runtime.cc
)

set(MLI_LIB_PUBLIC_INCLUDES
    $<BUILD_INTERFACE:${MLI_LIB_CMAKE_DIR}/../include>
    $<BUILD_INTERFACE:${MLI_LIB_CMAKE_DIR}/../include/internal>
    $<BUILD_INTERFACE:${MLI_LIB_CMAKE_DIR}/../include/api>
    $<BUILD_INTERFACE:${MLI_LIB_CMAKE_DIR}/src/private>
)

set(MLI_LIB_PRIVATE_INCLUDES
    ${MLI_LIB_CMAKE_DIR}/src/bricks
    ${MLI_LIB_CMAKE_DIR}/src/private
    ${MLI_LIB_CMAKE_DIR}/src/helpers
    ${MLI_LIB_CMAKE_DIR}/src/kernels
    ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling
    ${MLI_LIB_CMAKE_DIR}/src/kernels/conversion
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform
    ${MLI_LIB_CMAKE_DIR}/src/kernels/diverse
    ${MLI_LIB_CMAKE_DIR}/src/move
    ${MLI_LIB_CMAKE_DIR}/src/pal
)

set(MLI_LIB_PRIVATE_COMPILE_OPTIONS )

if (ARC)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
        -Hnocopyr
        -Hpurge
        -Hsdata0
        -Hdense_prologue
        -tcf_core_config
)
endif()

if (ARC)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
        -Werror
        -Wall
        -Wsign-compare
        -Wno-nonportable-include-path
    )
elseif (MSVC)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
            /W3
            /WX
        )
    else()
        # This path happens when other MSVC-commandline compatible
        # compilers are used like CLANG in Visual Studio.
    endif()
else()
    list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
        -Werror
    )
endif()

if (DEFINED MLI_BUILD_REFERENCE)
    set(choices
        ON
        OFF
    )
    if (NOT MLI_BUILD_REFERENCE IN_LIST choices)
        message(FATAL_ERROR "invalid MLI_BUILD_REFERENCE ${MLI_BUILD_REFERENCE}")
    endif()
    if (MLI_BUILD_REFERENCE STREQUAL "ON")
        list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
            MLI_BUILD_REFERENCE
        )
    endif()
endif()

if (DEFINED MLI_DBG_ENABLE_COMPILE_OPTION_MSG)
    set(choices
        ON
        OFF
    )
    if (NOT MLI_DBG_ENABLE_COMPILE_OPTION_MSG IN_LIST choices)
        message(FATAL_ERROR "invalid MLI_DBG_ENABLE_COMPILE_OPTION_MSG ${MLI_DBG_ENABLE_COMPILE_OPTION_MSG}")
    endif()
    if (MLI_DBG_ENABLE_COMPILE_OPTION_MSG STREQUAL "ON")
        list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
            MLI_DBG_ENABLE_COMPILE_OPTION_MSG
        )
    endif()
endif()

if (DEFINED MLI_DEBUG_MODE)
    set(choices
        DBG_MODE_RELEASE
        DBG_MODE_RET_CODES
        DBG_MODE_ASSERT
        DBG_MODE_DEBUG
        DBG_MODE_FULL
    )
    if (NOT MLI_DEBUG_MODE IN_LIST choices)
        message(FATAL_ERROR "invalid MLI_DEBUG_MODE ${MLI_DEBUG_MODE}")
    endif()
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        MLI_DEBUG_MODE=${MLI_DEBUG_MODE}
    )
endif()

# Supported values for rounding mode: UP/CONVERGENT (depends on platform)
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

if (NOT DEFINED FULL_ACCU)
    set(FULL_ACCU OFF)
endif()

if (NOT DEFINED AVEPOOL_16BIT_MUL)
    set(AVEPOOL_16BIT_MUL OFF)
endif()


if(ROUND_MODE STREQUAL UP)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_MODE_UP
    )
elseif(ROUND_MODE STREQUAL CONVERGENT)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_MODE_CONVERGENT
    )
else()
    message(FATAL_ERROR "rounding mode ${ROUND_MODE} is not supported")
endif()

if(FULL_ACCU STREQUAL ON)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        FULL_ACCU
    )
elseif(FULL_ACCU STREQUAL OFF)
    # we don't do anything in this case
else()
    message(FATAL_ERROR "Please specify full accumulator length: ON or OFF")
endif()

if(AVEPOOL_16BIT_MUL STREQUAL ON)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        AVEPOOL_16BIT_MUL
    )
elseif(AVEPOOL_16BIT_MUL STREQUAL OFF)
    # we don't do anything in this case
else()
    message(FATAL_ERROR "Please specify AVEPOOL_16BIT_MUL : ON or OFF")
endif()

if (${MLI_PLATFORM} STREQUAL VPX)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
            "SHELL: -mllvm -slot_swapping=true -mllvm -arc-vdsp-AA=1 -mllvm -no-stack-coloring")
    if(NOT ROUND_MODE STREQUAL UP)
        message(FATAL_ERROR "rounding mode ${ROUND_MODE} is not supported")
    endif()

elseif (${MLI_PLATFORM} STREQUAL EM_HS)
    if(ROUND_MODE STREQUAL CONVERGENT)
        list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
            -Xdsp_ctrl=postshift,guard,convergent
        )
    else()
        message(FATAL_ERROR "rounding mode ${ROUND_MODE} is not supported")
    endif()
endif()
