#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../cmake/settings.cmake)
include(mli_lib.cmake)

add_library(mli_lib STATIC ${MLI_LIB_SOURCE_FILES})

target_compile_definitions(mli_lib PUBLIC  ${MLI_LIB_PUBLIC_COMPILE_DEFINITIONS})
target_compile_definitions(mli_lib PRIVATE ${MLI_LIB_PRIVATE_COMPILE_DEFINITIONS})

target_include_directories(mli_lib PUBLIC  ${MLI_LIB_PUBLIC_INCLUDES})
target_include_directories(mli_lib PRIVATE ${MLI_LIB_PRIVATE_INCLUDES})

target_compile_options(mli_lib PRIVATE ${MLI_PLATFORM_COMPILE_OPTIONS})
target_compile_options(mli_lib PRIVATE ${MLI_LIB_PRIVATE_COMPILE_OPTIONS})

set_target_properties(mli_lib
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "../bin"
    LIBRARY_OUTPUT_DIRECTORY "../bin"
    RUNTIME_OUTPUT_DIRECTORY "../bin"
)
