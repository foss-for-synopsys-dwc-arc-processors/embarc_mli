#
# Copyright 2020-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../cmake/settings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/mli_build.cmake)
include(mli_lib.cmake)

set(project_name "mli")
set(project_version "3.0")

add_library(mli STATIC ${MLI_LIB_SOURCE_FILES})

target_compile_definitions(mli PUBLIC  ${MLI_LIB_PUBLIC_COMPILE_DEFINITIONS})
target_compile_definitions(mli PRIVATE ${MLI_LIB_PRIVATE_COMPILE_DEFINITIONS})

target_include_directories(mli PUBLIC
    $<INSTALL_INTERFACE:include/mli>
    $<INSTALL_INTERFACE:include/mli/api>
)
target_include_directories(mli PUBLIC  ${MLI_LIB_PUBLIC_INCLUDES})
target_include_directories(mli PRIVATE ${MLI_LIB_PRIVATE_INCLUDES})

target_compile_options(mli PRIVATE ${MLI_PLATFORM_COMPILE_OPTIONS})
target_compile_options(mli PRIVATE ${MLI_LIB_PRIVATE_COMPILE_OPTIONS})

if(ARC)
    # Subject to remove as the option in such form leads to warning on MSVC build.
    # set(CMAKE_CXX_STANDARD 17) is more toolchain agnostic and already present in 
    # the settings.cmake. But MWDT toolchain currently understands only the below format. 
    target_compile_options(mli PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()

set_target_properties(mli
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "../bin"
    LIBRARY_OUTPUT_DIRECTORY "../bin"
    RUNTIME_OUTPUT_DIRECTORY "../bin"
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../ CACHE PATH "..." FORCE)
endif()

file(GLOB MLI_INCLUDES
    "${MLI_LIB_HOME_DIR}/include/*.h"
    "${MLI_LIB_HOME_DIR}/include/*.hpp"
)

file(GLOB MLI_API_INCLUDES
    "${MLI_LIB_HOME_DIR}/include/api/*.h"
    "${MLI_LIB_HOME_DIR}/include/api/*.hpp"
)

install(
  FILES
    ${MLI_INCLUDES}
  DESTINATION
    include/mli)

install(
  FILES
    ${MLI_API_INCLUDES}
  DESTINATION
    include/mli/api)

export_library(mli)
