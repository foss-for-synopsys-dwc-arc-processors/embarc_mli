#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

DEBUG_BUILD         ?= ON
OPTMODE             ?= speed
GEN_EXAMPLES        ?= 1
MLI_BUILD_REFERENCE ?= OFF

include settings.mk

TOOL_CHAIN_OPTIONS =
ifndef TCF_FILE
ifeq ($(O_SYS),Windows)
TOOL_CHAIN_OPTIONS += -AWin32
endif
else
TOOL_CHAIN_OPTIONS += \
	-DARC_CFG_TCF_PATH=$(TCF_FILE) \
	-DCMAKE_TOOLCHAIN_FILE=$(subst \,$(PS),$(METAWARE_ROOT))$(PS)arc$(PS)cmake$(PS)arc-mwdt.toolchain.cmake \
	-G "Unix Makefiles"
ifdef BUILDLIB_DIR
TOOL_CHAIN_OPTIONS += -DBUILDLIB_DIR=$(BUILDLIB_DIR)
endif
endif

# $(MAKE) can be called through CMAKE; but to facilitate parallel builds with -j, we call $(MAKE).
ifndef TCF_FILE
lib: $(BUILD_DIR)/cmake_install.cmake
	cmake --build $(BUILD_DIR) --verbose  --target install
else
lib: $(BUILD_DIR)/cmake_install.cmake
	$(MAKE) -C $(BUILD_DIR) VERBOSE=true install 
endif

all: lib

$(BUILD_DIR)/cmake_install.cmake:
	cmake \
		-DDEBUG_BUILD=$(DEBUG_BUILD) \
		-DOPTMODE=$(OPTMODE) \
		-DGEN_EXAMPLES=$(GEN_EXAMPLES) \
		-DMLI_BUILD_REFERENCE=$(MLI_BUILD_REFERENCE) \
		-DCMAKE_INSTALL_PREFIX=${LIBRARY_DIR} \
		$(TOOL_CHAIN_OPTIONS) -B$(BUILD_DIR) -S..$(PS)..$(PS)

clean:
	-$(RMDIR) $(BUILD_DIR_ARC)
	-$(RMDIR) $(BUILD_DIR_NATIVE)
	-$(RMDIR) $(LIBRARY_DIR_ARC)
	-$(RMDIR) $(LIBRARY_DIR_NATIVE)

.PHONY:	all clean lib
