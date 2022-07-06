#
# Copyright 2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
cmake_policy(SET CMP0057 NEW)

# include_guard
if(__MLI_BUILD_CMAKE__)
  return()
endif()
set(__MLI_BUILD_CMAKE__ TRUE)

# WARNING: this file has similar functionality to build.cmake in build.git.
# Before making changes to this file, consult this file.
include("build" OPTIONAL RESULT_VARIABLE INCLUDE_BUILD_RESULT)
if (NOT ${INCLUDE_BUILD_RESULT} STREQUAL NOTFOUND)
  return()
endif()

# This option ensures that CMake will not use "-isystem" instead of "-I" when
# using find_package(). Without this option, some warnings may not show.
set (CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

macro(export_library_with_config config)
  if("" STREQUAL "${targets}")
    set(targets ${project_name})
  endif()

  install(
    TARGETS ${targets}
    EXPORT ${project_name}Targets
    DESTINATION lib)

  # Cannot use ARGN directly with list() command, so copy it to a variable
  # first.
  set(extra_macro_args ${ARGN})

  # Did we get any optional args?
  list(LENGTH extra_macro_args num_extra_args)
  if(${num_extra_args} GREATER 0)
    list(GET extra_macro_args 0 namespace)
  else()
    set(namespace ${project_name})
  endif()

  foreach(target ${targets})
    add_library(${namespace}::${target} ALIAS ${target})
  endforeach()

  install(
    EXPORT ${project_name}Targets
    FILE ${project_name}Targets.cmake
    NAMESPACE ${namespace}::
    DESTINATION lib/cmake/${project_name})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
    VERSION ${project_version}
    COMPATIBILITY SameMajorVersion)

  install(FILES ${config}
                ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
          DESTINATION lib/cmake/${project_name})
endmacro()

macro(export_library)
  export_library_with_config(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/${project_name}Config.cmake ${ARGN})
endmacro()

function(run)
  # not supported
endfunction()
