#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

project(har_smartphone)

include(../../cmake/settings.cmake)

file(GLOB temp
    ../auxiliary/*.cc
    ../auxiliary/*.c
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=8K
    -Hstack=1K
    -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Hdense_prologue
    -Hcl
    -Hcrt_argv
    -Hcrt_fast_memcpy
    -Hcrt_fast_memset
    -Hxcheck -Hcrt_initbss
)
else()
set(EXAMPLE_FLAGS)
endif()

function(ADD_DATA_TYPE DATA_TYPE DATA_TYPE_SHORT BIT_DEPTH)
    set(TARGET example_har_smartphone_${DATA_TYPE})
    add_executable(${TARGET}
        har_smartphone_coefficients_${DATA_TYPE_SHORT}.c
        har_smartphone_model_${DATA_TYPE_SHORT}.c
        ml_api_har_smartphone_main.c
        ${temp}
    )
    target_link_libraries(${TARGET} PUBLIC mli)

    target_include_directories(${TARGET} PRIVATE
        ../../examples/auxiliary
    )
 
    target_compile_options(${TARGET} PRIVATE
        ${MLI_PLATFORM_COMPILE_OPTIONS}
        ${EXAMPLE_FLAGS}
    )
    target_compile_definitions(${TARGET} PRIVATE
        MODEL_BIT_DEPTH=${BIT_DEPTH})
    target_link_options(${TARGET} PRIVATE
        ${MLI_PLATFORM_LINK_OPTIONS}
        ${EXAMPLE_FLAGS}
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

ADD_DATA_TYPE(FX16 fx 16)
ADD_DATA_TYPE(SA8 sa 8)
ADD_DATA_TYPE(FX16_FX8_FX8 fx 816)
