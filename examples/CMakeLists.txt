#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

project(mli_examples)

include(../cmake/settings.cmake)

add_subdirectory(hello_world)
add_subdirectory(example_cifar10_caffe)
add_subdirectory(example_har_smartphone)

# Example needs update
# add_subdirectory(example_face_detect)

if (ARC AND NOT (${MLI_PLATFORM} STREQUAL EM_HS) AND (NOT "$ENV{TENSORFLOW_DIR}" STREQUAL ""))
    add_subdirectory(tutorial_emnist_tflm)
endif()
