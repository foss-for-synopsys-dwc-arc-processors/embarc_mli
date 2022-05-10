#
# Copyright 2020-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

EMBARC_MLI_DIR := $(dir $(lastword $(MAKEFILE_LIST)))/../../..
include $(EMBARC_MLI_DIR)/make/rules.mk

BUILD_SUBDIR = examples$(PS)hello_world
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

BUILD_TARGET = hello_world

ifndef TCF_FILE
RUN_APP_CMD =
else
RUN_APP_CMD = mdb64 -cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif

app: build

run: app
	$(RUN_APP_CMD) $(BIN_PATH)$(PS)$(BUILD_TARGET)$(BIN_EXT)

.PHONY:	app run
