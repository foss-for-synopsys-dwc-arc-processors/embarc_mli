#
# Copyright 2020, Synopsys, Inc.
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

if (NOT DEFINED MLI_LIB_HOME_DIR)
    message(FATAL_ERROR "Please define MLI_LIB_HOME_DIR")
endif()
set(MLI_LIB_CMAKE_DIR ${MLI_LIB_HOME_DIR}/lib)

include(${MLI_LIB_CMAKE_DIR}/../cmake/settings.cmake)

# To keep code similar to our make files, we use file(GLOB...) to add source files, consider to explicitly add them.
if (ARC AND (${MLI_PLATFORM} STREQUAL EM_HS))

    file(GLOB temp
        ${MLI_LIB_CMAKE_DIR}/src/helpers/src/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/common/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/data_manip/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling_chw/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/private/src/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/bricks/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/move/*.cc
    )
    set(MLI_LIB_SOURCE_FILES
        ${temp}
    )

else()
    file(GLOB temp
        ${MLI_LIB_CMAKE_DIR}/src/helpers/src/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling/*hwc*.cc
        ${MLI_LIB_CMAKE_DIR}/src/bricks/*.cc
        ${MLI_LIB_CMAKE_DIR}/src/private/src/*.cc
    )
    set(MLI_LIB_SOURCE_FILES
        ${temp}
        ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_relu_fx.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_sigm_fx.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_tanh_fx.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/transform/mli_krn_softmax_fx.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_conv2d_hwcn.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/convolution/mli_krn_depthwise_conv2d_hwcn.cc
        ${MLI_LIB_CMAKE_DIR}/src/kernels/common/mli_krn_fully_connected.cc
    )
endif()

set(MLI_LIB_PUBLIC_INCLUDES
    ${MLI_LIB_CMAKE_DIR}/../include
    ${MLI_LIB_CMAKE_DIR}/../include/api
    ${MLI_LIB_CMAKE_DIR}/../lib/src/private
)

set(MLI_LIB_PRIVATE_INCLUDES
    ${MLI_LIB_CMAKE_DIR}/src/bricks
    ${MLI_LIB_CMAKE_DIR}/src/private
    ${MLI_LIB_CMAKE_DIR}/src/helpers
    ${MLI_LIB_CMAKE_DIR}/src/kernels
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling
    ${MLI_LIB_CMAKE_DIR}/src/kernels/pooling_chw
    ${MLI_LIB_CMAKE_DIR}/src/kernels/transform
    ${MLI_LIB_CMAKE_DIR}/src/move
    ${MLI_LIB_CMAKE_DIR}/src/pal
)

if (ARC)
    set(MLI_LIB_PRIVATE_COMPILE_OPTIONS
        -Hnocopyr
        -Hpurge
        -Hsdata0
        -Hdense_prologue
        -Wall
        -Wno-nonportable-include-path
    )
elseif (MSVC)
    set(MLI_LIB_PRIVATE_COMPILE_OPTIONS
        /W3
    )
else()
    set(MLI_LIB_PRIVATE_COMPILE_OPTIONS
        -Werror
        -Wno-nonportable-include-path
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
    else()
        message(FATAL_ERROR "Please specify a rounding mode: UP or CONVERGENT")
    endif()
endif()

if(ROUND_MODE STREQUAL UP)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_UP
    )
elseif(ROUND_MODE STREQUAL CONVERGENT)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_CONVERGENT
    )
else()
    message(FATAL_ERROR "rounding mode ${ROUND_MODE} is not supported")
endif()

if (${MLI_PLATFORM} STREQUAL VPX)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
        "SHELL: -mllvm -slot_swapping=true")

elseif (${MLI_PLATFORM} STREQUAL EM_HS)
    if(ROUND_MODE STREQUAL UP)
        list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
            -Xdsp_ctrl=postshift,guard,up
        )
    elseif(ROUND_MODE STREQUAL CONVERGENT)
        list(APPEND MLI_LIB_PRIVATE_COMPILE_OPTIONS
            -Xdsp_ctrl=postshift,guard,convergent
        )
    else()
        message(FATAL_ERROR "rounding mode ${ROUND_MODE} is not supported")
    endif()
endif()
