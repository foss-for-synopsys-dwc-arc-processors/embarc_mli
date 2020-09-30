#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../cmake/settings.cmake)

file(GLOB test_components_src
    test_components/*.cc
    test_components/*.c
    ../examples/auxiliary/*.cc
    ../examples/auxiliary/*.c
)

set(test_components_inc
    test_components
    ../examples/auxiliary
)

if (ARC)
set(TEST_COMPILE_OPTIONS
    -Hnocopyr
    -Hpurge
    -Hheap=16K
    -Hstack=16K
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

#======================================================
# Convolution 2D Tests
#======================================================
add_executable(test_mli_krn_conv2d
    tests/mli_krn_conv2d/tests_mli_krn_conv2d.cc
    ${test_components_src}
)

target_include_directories(test_mli_krn_conv2d PRIVATE
    tests/mli_krn_conv2d
    ${test_components_inc}
)

target_link_libraries(test_mli_krn_conv2d PUBLIC mli)

target_compile_options(test_mli_krn_conv2d PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
)

target_link_options(test_mli_krn_conv2d PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(test_mli_krn_conv2d PRIVATE
    -m
    -Coutput=./bin/test_mli_krn_conv2d.map
)
endif()

set_target_properties(test_mli_krn_conv2d
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
)

#======================================================
# Softmax
#======================================================
add_executable(test_mli_krn_softmax
    tests/mli_krn_softmax/tests_mli_krn_softmax.cc
    ${test_components_src}
)

target_include_directories(test_mli_krn_softmax PRIVATE
    tests/mli_krn_softmax
    ${test_components_inc}
)

target_link_libraries(test_mli_krn_softmax PUBLIC mli)

target_compile_options(test_mli_krn_softmax PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
)

target_link_options(test_mli_krn_softmax PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${TEST_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(test_mli_krn_softmax PRIVATE
    -m
    -Coutput=./bin/test_mli_krn_softmax.map
)
endif()

set_target_properties(test_mli_krn_softmax
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
)

