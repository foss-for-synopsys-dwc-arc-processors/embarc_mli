#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

if(_MLI_SETTINGS_CMAKE_LOADED)
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

if (ARC AND (${MLI_PLATFORM} STREQUAL VPX))
set(MLI_PLATFORM_COMPILE_OPTIONS
   -O3
   -Hvdsp_vector_c
)
set(MLI_PLATFORM_LINK_OPTIONS
    -Hnocopyr
    -Hpurge
    -Hheap=1024K
    -Hstack=8K
    -Hon=Long_enums
    -Hvdsp_vector_c
    -Hlib=${BUILDLIB_DIR}
)

elseif (ARC AND (${MLI_PLATFORM} STREQUAL EM_HS))
set(MLI_PLATFORM_COMPILE_OPTIONS
   -O3
   -Xdsp_ctrl=postshift,guard,convergent
)
set(MLI_PLATFORM_LINK_OPTIONS
    -Hnocopyr
    -Hpurge
    -Hheap=1024K
    -Hstack=8K
    -Hon=Long_enums
    -Hfxapi
)

elseif ((NOT ARC) AND (NOT MSVC))
set(MLI_PLATFORM_COMPILE_OPTIONS
    -m32
)
set(MLI_PLATFORM_LINK_OPTIONS
    -m32
)

endif()
