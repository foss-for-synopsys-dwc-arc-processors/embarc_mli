#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

if(_MLI_LIB_CMAKE_LOADED)
  return()
endif()
set(_MLI_LIB_CMAKE_LOADED TRUE)

function(get_path_to_mli_lib_cmake MLI_LIB_CMAKE_DIR)
    set(${MLI_LIB_CMAKE_DIR} ${CMAKE_CURRENT_FUNCTION_LIST_DIR} PARENT_SCOPE)
endfunction()
get_path_to_mli_lib_cmake(MLI_LIB_CMAKE_DIR)

set(MLI_LIB_SOURCE_FILES
    ${MLI_LIB_CMAKE_DIR}/src/helpers/src/mli_helpers.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/mli_krn_eltwise_add_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/mli_krn_eltwise_max_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/mli_krn_eltwise_min_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/mli_krn_eltwise_mul_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/kernels/eltwise/mli_krn_eltwise_sub_fx.cc
    ${MLI_LIB_CMAKE_DIR}/src/private/src/mli_check.cc
)

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
    ${MLI_LIB_CMAKE_DIR}/../examples/auxiliary
)

if(MSVC)
set(MLI_LIB_PRIVATE_COMPILE_OPTIONS
    /W3)
else()
set(MLI_LIB_PRIVATE_COMPILE_OPTIONS
    -Werror
    -Wno-nonportable-include-path
)
endif()

list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
    MLI_BUILD_REFERENCE)

if(NOT DEFINED ROUND_MODE)
    if(${MLI_PLATFORM} STREQUAL VPX)
        set(ROUND_MODE UP)
    elseif(${MLI_PLATFORM} STREQUAL EM_HS)
        set(ROUND_MODE CONVERGENT)
    else()
        set(ROUND_MODE CONVERGENT)
    endif()
endif()

if(ROUND_MODE STREQUAL CONVERGENT)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_CONVERGENT)
elseif(ROUND_MODE STREQUAL UP)
    list(APPEND MLI_LIB_PRIVATE_COMPILE_DEFINITIONS
        ROUND_UP)
else()
    message(FATAL_ERROR "Rounding mode isn't supported!")
endif()
