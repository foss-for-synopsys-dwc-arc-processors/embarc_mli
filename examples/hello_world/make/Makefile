#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# Not providing a TCF_FILE tries to compile with the native compiler on the host platform.
# Provide a build lib for those targets that need.
# BUILDLIB_DIR?=ev71_buildlib
# TCF_FILE=$(CURDIR)/../../../hw/em9d.tcf

include ../../../lib/make/settings.mk
PUBLIC_DIR = ..$(PS)..$(PS)..$(PS)

# Targets
app:
	$(MAKE) -C $(PUBLIC_DIR)lib/make lib
	cmake --build $(BUILD_DIR)$(PS)examples

clean:
	$(MAKE) -C $(PUBLIC_DIR)lib/make clean

ifndef TCF_FILE
run: app
	$(BUILD_DIR)$(PS)examples$(PS)hello_world$(PS)elf_files$(PS)hello_world
else
run: app
	mdb $(BUILD_DIR)$(PS)examples$(PS)hello_world$(PS)elf_files$(PS)hello_world \
		-cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
endif

.PHONY:	app clean run
