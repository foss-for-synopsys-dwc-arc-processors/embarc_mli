#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../../cmake/settings.cmake)

add_executable(example_face_detect
    bmp_file_io.c
    face_trigger_model.c
    main.c
    sliding_scan.c
)

target_link_libraries(example_face_detect PUBLIC mli)

target_include_directories(example_face_detect PRIVATE
    ../../examples/auxiliary
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=20K
    -Hstack=4K
    -Hfxapi
    -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Xdsp_ctrl=postshift,guard,convergent
    -Hdense_prologue
)
else()
set(EXAMPLE_FLAGS)
endif()

target_compile_options(example_face_detect PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

target_link_options(example_face_detect PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(example_face_detect PRIVATE
    -m
    -Coutput=./elf_files/example_face_detect.map
)
endif()

set_target_properties(example_face_detect
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./elf_files$<0:>"
)
