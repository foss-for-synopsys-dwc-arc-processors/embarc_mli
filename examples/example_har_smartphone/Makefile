#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# TCF_FILE: withouts NOT setting this parameter,
#           the default native compiler is used to compile
# BUILDLIB_DIR ?= ev71_buildlib can be specified if no MWDT buildlib is available
# See rules.mk for all flags

PUBLIC_DIR = ../..
include $(PUBLIC_DIR)/make/rules.mk

BUILD_SUBDIR = examples$(PS)example_har_smartphone
BUILD_TARGET = example_har_smartphone

run: app
ifndef TCF_FILE
	$(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)elf_files$(PS)example_har_smartphone
else
	mdb $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)elf_files$(PS)example_har_smartphone.elf \
		-cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE) -- $(RUN_ARGS)
endif

.PHONY:	app run
