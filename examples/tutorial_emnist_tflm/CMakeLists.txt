#
# Copyright 2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

project(tutorial_emnist_tflm)

include(../../cmake/settings.cmake)

if (ARC)
    set(EXAMPLE_FLAGS
        -Hheap=2K
        -Hstack=5K
        -Hnocopyr
        -Hpurge
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

link_directories(./third_party/)

set(TARGET tutorial_emnist_tflm)
add_executable(${TARGET}
    ./src/main.cc
    ./src/model_settings.cc
    ./src/test_samples.cc
)

target_link_libraries(${TARGET} PUBLIC $ENV{TENSORFLOW_DIR}/tensorflow/lite/micro/tools/make/downloads/arc_mli_$ENV{TCF_FILE_NAME}_mli20/bin/arc/libmli.a)
target_link_libraries(${TARGET} PRIVATE tensorflow-microlite)

target_include_directories(${TARGET} PRIVATE
    $ENV{PUBLIC_DIR}/include
    $ENV{PUBLIC_DIR}/include/api
    $ENV{TENSORFLOW_DIR}
    $ENV{TENSORFLOW_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
    $ENV{TENSORFLOW_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
)
target_compile_options(${TARGET} PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)
target_compile_definitions(${TARGET} PRIVATE
    TF_LITE_STATIC_MEMORY
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

