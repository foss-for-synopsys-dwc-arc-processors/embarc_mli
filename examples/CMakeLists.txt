#
# Copyright 2020, Synopsys, Inc.
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
add_subdirectory(tutorial_emnist_tensorflow)

if (ARC AND (${MLI_PLATFORM} STREQUAL EM_HS))
    add_subdirectory(example_face_detect)
endif()