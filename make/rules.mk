#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
$(error "Requires make version 3.82 or later (current is $(MAKE_VERSION))")
endif

RECONFIGURE         ?= OFF
DEBUG_BUILD         ?= ON
OPTMODE             ?= speed
GEN_EXAMPLES        ?= 1
MLI_BUILD_REFERENCE ?= OFF
BUILD_SUBDIR        ?=
BUILD_TARGET        ?= install
EXT_CFLAGS          ?=
CMAKE_OPTIONS       ?=
# User need to define BUILD_LIB (target specific libs including runtime)
# if default one doesn't fit.
BUILDLIB_DIR        ?=
PERFORM_BUILD       ?= ON
VERBOSE             ?= OFF

include $(PUBLIC_DIR)/make/settings.mk

TOOLCHAIN_OPTIONS = $(CMAKE_OPTIONS)
ifdef TCF_FILE
# For TCF_FILE, we only take the realpath if the file exists (otherwise it is a file inside MWDT)
TOOLCHAIN_OPTIONS += \
	-DARC_CFG_TCF_PATH=$(if $(realpath $(TCF_FILE)),$(realpath $(TCF_FILE)),$(TCF_FILE)) \
	-DCMAKE_TOOLCHAIN_FILE=$(abspath $(METAWARE_ROOT)$(PS)arc$(PS)cmake$(PS)arc-mwdt.toolchain.cmake) \
	-G "Unix Makefiles"
ifdef BUILDLIB_DIR
# For BUILDLIB_DIR, we only take the realpath if the dir exists (otherwise it is a dir inside MWDT)
TOOLCHAIN_OPTIONS += -DBUILDLIB_DIR=$(if $(realpath $(BUILDLIB_DIR)),$(realpath $(BUILDLIB_DIR)),$(BUILDLIB_DIR))
endif
endif

ifdef ROUND_MODE
TOOLCHAIN_OPTIONS += -DROUND_MODE=${ROUND_MODE}
endif

ifdef FULL_ACCU
TOOLCHAIN_OPTIONS += -DFULL_ACCU=${FULL_ACCU}
endif

ifdef MLI_DEBUG_MODE
TOOLCHAIN_OPTIONS += -DMLI_DEBUG_MODE=${MLI_DEBUG_MODE}
endif

ifdef MLI_DBG_ENABLE_COMPILE_OPTION_MSG
TOOLCHAIN_OPTIONS += -DMLI_DBG_ENABLE_COMPILE_OPTION_MSG=${MLI_DBG_ENABLE_COMPILE_OPTION_MSG}
endif

ifeq ($(RECONFIGURE),ON)
CONFIG_TARGET=config
else
CONFIG_TARGET=$(BUILD_DIR)/cmake_install.cmake
endif

ifdef JOBS
JOBS_OPTIONS=-j$(JOBS)
endif

lib: $(CONFIG_TARGET)
ifeq ($(PERFORM_BUILD),ON)
	cmake --build $(abspath $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)) --target install $(JOBS_OPTIONS)
endif

build: $(CONFIG_TARGET)
ifeq ($(PERFORM_BUILD),ON)
	cmake --build $(abspath $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)) --target $(BUILD_TARGET) $(JOBS_OPTIONS)
endif

ifeq ($(VERBOSE),ON)
export VERBOSE
else
unexport VERBOSE
endif

$(CONFIG_TARGET):
	cmake \
		-DDEBUG_BUILD=$(DEBUG_BUILD) \
		-DEXT_CFLAGS=$(EXT_CFLAGS) \
		-DOPTMODE=$(OPTMODE) \
		-DGEN_EXAMPLES=$(GEN_EXAMPLES) \
		-DMLI_BUILD_REFERENCE=$(MLI_BUILD_REFERENCE) \
		-DCMAKE_INSTALL_PREFIX=$(abspath $(LIBRARY_DIR)) \
		$(TOOLCHAIN_OPTIONS) \
		-B$(abspath $(BUILD_DIR)) \
		-S$(abspath $(PUBLIC_DIR))

cleanall:
# quotes are required if path contains slashes on Windows
	-$(RMDIR) "$(BUILD_DIR_BASE)"
	-$(RMDIR) "$(LIBRARY_DIR_BASE)"

clean:
# Check that cmake generated a Makefile project. 
# In this case we have an extra assumption to make a more tidy clean 
# rule (remove only application objects/binaries).  Otherwise, it may not work 
# for several project types (as MSVC) and requires cleanin all solution
ifneq ($(realpath $(BUILD_DIR)$(PS)Makefile),)
	cmake --build $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) --target clean
else
	cmake --build $(BUILD_DIR) --target clean
endif

.PHONY:	clean cleanall lib build config
