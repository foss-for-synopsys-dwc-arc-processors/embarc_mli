#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.12)

project(mli)

add_subdirectory(lib)

add_subdirectory(user_tests)

if (GEN_EXAMPLES)
	add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()