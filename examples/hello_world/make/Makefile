#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

#TCF_FILE=$(CURDIR)/../../../hw/em9d.tcf
# Not providing a TCF_FILE tries to compile with the native compiler on the host platform.
# Provide a build lib for those targets that need:
# BUILDLIB_DIR?=ev71_buildlib

# From rules.mk
BACKSLASH=\$(nullstring)
ifneq ($(ComSpec)$(COMSPEC),)
	O_SYS=Windows
	RM=del /F /Q
	RMDIR=rmdir /Q /S
	MKDIR=mkdir
	PS=$(BACKSLASH)
else
	O_SYS=Linux
	RM=rm -rf
	RMDIR=rm -rf
	MKDIR=mkdir -p
	PS=/
endif

BUILD_FOLDER_ARC = ..$(PS)..$(PS)..$(PS)outputs_arc
BUILD_FOLDER_NATIVE = ..$(PS)..$(PS)..$(PS)outputs_native

# Set target specific setting for NATIVE build
ifndef TCF_FILE
BUILD_FOLDER=$(BUILD_FOLDER_NATIVE)
ifeq ($(UNAME),Windows)
TOOL_CHAIN_OPTIONS = -AWin32
else
TOOL_CHAIN_OPTIONS =
endif
else
BUILD_FOLDER=$(BUILD_FOLDER_ARC)
TOOL_CHAIN_OPTIONS = \
	-DARC_CFG_TCF_PATH=$(TCF_FILE) \
	-DCMAKE_TOOLCHAIN_FILE=$(subst \,$(PS),$(METAWARE_ROOT))$(PS)arc$(PS)cmake$(PS)arc-mwdt.toolchain.cmake \
	-G "Unix Makefiles"
ifdef BUILDLIB_DIR
TOOL_CHAIN_OPTIONS += -DBUILDLIB_DIR=$(BUILDLIB_DIR)
endif
endif

# Targets
$(BUILD_FOLDER)/cmake_install.cmake:
	echo CONFIGURING $(TARGET)
	cmake -DGEN_EXAMPLES=1 $(TOOL_CHAIN_OPTIONS) -B$(BUILD_FOLDER) -S..$(PS)..$(PS)..$(PS)

app: $(BUILD_FOLDER)/cmake_install.cmake
	echo BUILDING $(TARGET)
	cmake --build $(BUILD_FOLDER) --verbose

clean:
	-$(RMDIR) $(BUILD_FOLDER_ARC)
	-$(RMDIR) $(BUILD_FOLDER_NATIVE)

ifndef TCF_FILE
run: app
	$(BUILD_FOLDER)$(PS)examples$(PS)hello_world$(PS)elf_files$(PS)hello_world
else
run: app
	mdb $(BUILD_FOLDER)$(PS)examples$(PS)hello_world$(PS)elf_files$(PS)hello_world \
		-cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
endif

.PHONY:	app clean
