#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# This function sets common flags for MLI compilation:
#   MLI_PLATFORM
#   MLI_PLATFORM_LINK_OPTIONS
#   MLI_PLATFORM_COMPILE_OPTIONS
# FLAGS here are similar to build/rules.mk, notable excpetions:
#   MLI_PLATFORM: is set here
#   -Hvdsp_vector_c / -Hfxapi is set here.

if (_MLI_SETTINGS_CMAKE_LOADED)
  return()
endif()
set(_MLI_SETTINGS_CMAKE_LOADED TRUE)

function(get_mli_platform MLI_PLATFORM)
    if (DEFINED ARC_CFG_TCF_PATH)
        execute_process (
            COMMAND ccac -tcf=${ARC_CFG_TCF_PATH} -Hbatchnotmp _.c
            OUTPUT_VARIABLE outVar
        )
        string(FIND ${outVar} "+vdsp" found_vdsp)
        if (${found_vdsp} GREATER -1)
            set(MLI_PLATFORM VPX PARENT_SCOPE)
        else()
            set(MLI_PLATFORM EM_HS PARENT_SCOPE)
        endif()
    else()
        set(MLI_PLATFORM NATIVE PARENT_SCOPE)
    endif()
    message(STATUS ${MLI_PLATFORM})
endfunction()

get_mli_platform(MLI_PLATFORM)

set(MLI_PLATFORM_LINK_OPTIONS)
set(MLI_PLATFORM_COMPILE_OPTIONS)
set(MLI_PLATFORM_FLAGS)

if (ARC)
    if (NOT DEFINED DEBUG_BUILD)
        set(DEBUG_BUILD ON)
    endif()
    if (DEBUG_BUILD STREQUAL ON)
        list(APPEND MLI_PLATFORM_FLAGS
            -g
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
        -Hon=Long_enums
        "SHELL: -mllvm -gen-lpcc=false -mllvm -arc-scev-aa-vdsp=false"
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
endif()

list(APPEND MLI_PLATFORM_COMPILE_OPTIONS ${MLI_PLATFORM_FLAGS})
list(APPEND MLI_PLATFORM_LINK_OPTIONS    ${MLI_PLATFORM_FLAGS})

