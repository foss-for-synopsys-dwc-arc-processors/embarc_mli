#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../../cmake/settings.cmake)

add_executable(hello_world
    main.cpp
)

target_link_libraries(hello_world PUBLIC mli)

target_compile_options(hello_world PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
)

target_link_options(hello_world PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
)

if (ARC)
target_link_options(hello_world PRIVATE
    -m
    -Coutput=./bin/hello_world.map
    -Hheap=4K
    -Hstack=4K
)
endif()

set_target_properties(hello_world
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./bin$<0:>"
)

SET(CMAKE_DEFAULT_STARTUP_PROJECT hello_world)
