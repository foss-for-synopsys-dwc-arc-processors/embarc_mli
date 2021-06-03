#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

project(face_detect)

include(../../cmake/settings.cmake)

set(TARGET example_face_detect)

add_executable(${TARGET}
    bmp_file_io.c
    bmp_file_io.h
    face_detect_module.cc
    face_detect_module.h
    main.cc
    model/anchors.cc
    model/defs.h
    model/model.cc
    model/model.h
    model/postprocess.cc
    model/postprocess.h
    model/util.cc
    model/util.h
    model/weights.cc
    model/weights.h
)

target_link_libraries(${TARGET} PUBLIC mli)

target_include_directories(${TARGET} PRIVATE
    ../../examples/auxiliary
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=50K
    -Hstack=8K
    -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Hdense_prologue
    -Hcl
    -Hcrt_argv
    -Hcrt_fast_memcpy
    -Hcrt_fast_memset
    -Hxcheck
    -Hcrt_initbss

)
else()
set(EXAMPLE_FLAGS)
endif()

target_compile_options(${TARGET} PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

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
