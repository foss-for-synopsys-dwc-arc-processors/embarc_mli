#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

RECONFIGURE         ?= OFF
DEBUG_BUILD         ?= ON
OPTMODE             ?= speed
GEN_EXAMPLES        ?= 1
MLI_BUILD_REFERENCE ?= OFF
BUILD_SUBDIR        ?=
EXT_CFLAGS          ?=
TOOLCHAIN_OPTIONS   ?=

include settings.mk

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

ifdef MLI_DEBUG_MODE
TOOLCHAIN_OPTIONS += -DMLI_DEBUG_MODE=${MLI_DEBUG_MODE}
endif

ifeq ($(RECONFIGURE),ON)
CONFIG_TARGET=config
else
CONFIG_TARGET=$(BUILD_DIR)/cmake_install.cmake
endif

# $(MAKE) can be called through CMAKE; but to facilitate parallel builds with -j, we call $(MAKE).
ifndef TCF_FILE
lib: $(CONFIG_TARGET)
	cmake --build $(abspath $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)) --verbose --target install
else
lib: $(CONFIG_TARGET)
	$(MAKE) -C $(abspath $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)) VERBOSE=true install
endif

all: lib

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
		-S$(abspath ..$(PS)..$(PS))

clean:
	-$(RMDIR) $(BUILD_DIR_BASE)
	-$(RMDIR) $(LIBRARY_DIR_BASE)

.PHONY:	all clean lib config
