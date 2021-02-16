#
# Copyright 2020-2021, Synopsys, Inc.
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
add_executable(example_cifar10_caffe
    cifar10_coefficients_hwcn_small.c
    cifar10_model_hwcn.c
    ml_api_cifar10_caffe_main.c
    ${temp}
)

target_link_libraries(example_cifar10_caffe PUBLIC mli)

target_include_directories(example_cifar10_caffe PRIVATE
    ../../examples/auxiliary
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=8K
    -Hstack=1K
    -Hfxapi -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Xdsp_ctrl=postshift,guard,convergent
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

target_compile_options(example_cifar10_caffe PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

target_link_options(example_cifar10_caffe PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(example_cifar10_caffe PRIVATE
    -m
    -Coutput=./elf_files/example_cifar10_caffe.map
)
endif()

set_target_properties(example_cifar10_caffe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./elf_files$<0:>"
)
