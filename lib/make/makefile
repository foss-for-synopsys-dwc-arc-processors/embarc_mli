#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

RECONFIGURE         ?= ON
DEBUG_BUILD         ?= ON
OPTMODE             ?= speed
GEN_EXAMPLES        ?= 1
MLI_BUILD_REFERENCE ?= OFF
BUILD_SUBDIR        ?=

include settings.mk

TOOL_CHAIN_OPTIONS =
ifndef TCF_FILE
ifeq ($(O_SYS),Windows)
TOOL_CHAIN_OPTIONS += -AWin32
endif
else
TOOL_CHAIN_OPTIONS += \
	-DARC_CFG_TCF_PATH=$(abspath $(TCF_FILE)) \
	-DCMAKE_TOOLCHAIN_FILE=$(abspath $(METAWARE_ROOT)$(PS)arc$(PS)cmake$(PS)arc-mwdt.toolchain.cmake) \
	-G "Unix Makefiles"
ifdef BUILDLIB_DIR
TOOL_CHAIN_OPTIONS += -DBUILDLIB_DIR=$(abspath $(BUILDLIB_DIR))
endif
endif

ifdef ROUND_MODE
TOOL_CHAIN_OPTIONS += -DROUND_MODE=${ROUND_MODE}
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
		-DOPTMODE=$(OPTMODE) \
		-DGEN_EXAMPLES=$(GEN_EXAMPLES) \
		-DMLI_BUILD_REFERENCE=$(MLI_BUILD_REFERENCE) \
		-DCMAKE_INSTALL_PREFIX=$(abspath $(LIBRARY_DIR)) \
		$(TOOL_CHAIN_OPTIONS) \
		-B$(abspath $(BUILD_DIR)) \
		-S$(abspath ..$(PS)..$(PS))

clean:
	-$(RMDIR) $(BUILD_DIR_BASE)
	-$(RMDIR) $(LIBRARY_DIR_BASE)

.PHONY:	all clean lib config
