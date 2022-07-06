#
# Copyright 2020-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# This function sets common flags for MLI compilation:
#   MLI_PLATFORM
#   MLI_PLATFORM_LINK_OPTIONS
#   MLI_PLATFORM_COMPILE_OPTIONS

if (_MLI_SETTINGS_CMAKE_LOADED)
    return()
endif()
set(_MLI_SETTINGS_CMAKE_LOADED TRUE)

# Workaround to handle differences in used build systems.
if(DEFINED BUILD_DEVICE_ARC AND DEFINED ARC)
    message(FATAL_ERROR "Both BUILD_DEVICE_ARC and ARC are set, use only 1 when calling the top-level CMake.")
endif()
# Make sure BUILD_DEVICE_ARC and ARC are qual to each other.
if(DEFINED BUILD_DEVICE_ARC)
    set(ARC ON)
endif()
if(DEFINED ARC)
    set(BUILD_DEVICE_ARC ON)
endif()

# Workaround to handle differences in used build systems.
if(DEFINED EVSS_CFG_TCF_PATH AND DEFINED ARC_CFG_TCF_PATH)
    message(FATAL_ERROR "Both EVSS_CFG_TCF_PATH and ARC_CFG_TCF_PATH are set, use only 1 when calling the top-level CMake.")
endif()
# Make sure ARC_CFG_TCF_PATH and EVSS_CFG_TCF_PATH are qual to each other.
if(DEFINED ARC_CFG_TCF_PATH)
    set(EVSS_CFG_TCF_PATH ${ARC_CFG_TCF_PATH})
endif()
if(DEFINED EVSS_CFG_TCF_PATH)
    set(ARC_CFG_TCF_PATH ${EVSS_CFG_TCF_PATH})
endif()

# Query the compiler to get more information about the ARC platform
function(get_mli_platform MLI_PLATFORM)
    if (DEFINED ARC_CFG_TCF_PATH)
        execute_process (
            COMMAND ccac -tcf=${ARC_CFG_TCF_PATH} -Hbatchnotmp _.c
            OUTPUT_VARIABLE outVar
        )
        string(FIND ${outVar} "+vdsp" found_vdsp)
        if (${found_vdsp} GREATER -1)
            set(${MLI_PLATFORM} VPX PARENT_SCOPE)
        else()
            set(${MLI_PLATFORM} EM_HS PARENT_SCOPE)
        endif()
    else()
        set(${MLI_PLATFORM} NATIVE PARENT_SCOPE)
    endif()
endfunction()

get_mli_platform(MLI_PLATFORM)
message(STATUS "${MLI_PLATFORM}")

set(MLI_PLATFORM_LINK_OPTIONS)
set(MLI_PLATFORM_COMPILE_OPTIONS)
set(MLI_PLATFORM_FLAGS)
set(CMAKE_CXX_STANDARD 17)

if (ARC)
    if (NOT DEFINED DEBUG_BUILD)
        set(DEBUG_BUILD ON)
    endif()
    if (DEBUG_BUILD STREQUAL ON)
        list(APPEND MLI_PLATFORM_FLAGS
            -g
            -O0
            -fstack-protector-all
        )
    endif()

    if (NOT DEFINED OPTMODE)
        set(OPTMODE speed)
    endif()

    if (OPTMODE STREQUAL size)
        list(APPEND MLI_PLATFORM_FLAGS
            -O2
            -Hlto
        )
    elseif (OPTMODE STREQUAL speed)
        list(APPEND MLI_PLATFORM_FLAGS
            -O3
        )
    else()
        message(FATAL_ERROR invalid OPTMODE)
    endif()

    list(APPEND MLI_PLATFORM_FLAGS
        -Wcg,-arc-vdsp-AA=1
        "SHELL: -mllvm -gen-lpcc=false -mllvm -arc-sort-out-copy=true -mllvm -arc-vdsp-copy=3"
    )
    if (DEFINED BUILDLIB_DIR)
        list(APPEND MLI_PLATFORM_LINK_OPTIONS
            -Hlib=${BUILDLIB_DIR}
        )
    endif()

    if (${MLI_PLATFORM} STREQUAL VPX)
        list(APPEND MLI_PLATFORM_FLAGS
            -Hvdsp_vector_c
        )
    elseif (${MLI_PLATFORM} STREQUAL EM_HS)
        list(APPEND MLI_PLATFORM_FLAGS
            -Hfxapi
        )
    endif()

    set(CMAKE_EXECUTABLE_SUFFIX .elf)
elseif (MSVC)
    if (NOT DEFINED DEBUG_BUILD)
        set(DEBUG_BUILD ON)
    endif()
    if (DEBUG_BUILD STREQUAL ON)
        list(APPEND MLI_PLATFORM_FLAGS
            /Zi
            /Od
            /GS
        )

    if (NOT DEFINED MLI_DEBUG_MODE)
        list(APPEND MLI_PLATFORM_FLAGS /DMLI_DEBUG_MODE=DBG_MODE_RELEASE)
    else()
        list(APPEND MLI_PLATFORM_FLAGS /DMLI_DEBUG_MODE=${MLI_DEBUG_MODE})
    endif()
else()
    if (NOT DEFINED DEBUG_BUILD)
        set(DEBUG_BUILD ON)
    endif()
    if (DEBUG_BUILD STREQUAL ON)
        list(APPEND MLI_PLATFORM_FLAGS
            -g
            -O0
            -fstack-protector-all
        )
    endif()
endif()
endif()

list(APPEND MLI_PLATFORM_COMPILE_OPTIONS ${MLI_PLATFORM_FLAGS})
if (MSVC)
else()
    list(APPEND MLI_PLATFORM_LINK_OPTIONS ${MLI_PLATFORM_FLAGS})
endif()
