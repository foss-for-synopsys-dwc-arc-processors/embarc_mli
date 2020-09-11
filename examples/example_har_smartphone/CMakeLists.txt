#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../../cmake/settings.cmake)

file(GLOB temp
    ../auxiliary/*.cc
    ../auxiliary/*.c
)
add_executable(example_har_smartphone
    har_smartphone_coefficients.c
    har_smartphone_model.c
    ml_api_har_smartphone_main.c
    ${temp}
)

target_link_libraries(example_har_smartphone PUBLIC mli)

target_include_directories(example_har_smartphone PRIVATE
    ../../examples/auxiliary
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=8K
    -Hstack=1K
    -Hfxapi
    -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Xdsp_ctrl=postshift,guard,convergent
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

target_compile_options(example_har_smartphone PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

target_link_options(example_har_smartphone PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(example_har_smartphone PRIVATE
    -m
    -Coutput=./elf_files/example_har_smartphone.map
)
endif()

set_target_properties(example_har_smartphone
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./elf_files$<0:>"
)

