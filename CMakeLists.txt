#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.17)

# set global flags for the compiler; only set EXT_CFLAGS
# if there are no other methods available.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXT_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXT_CFLAGS}")

project(mli)

add_subdirectory(lib)

if (GEN_EXAMPLES)
	add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()