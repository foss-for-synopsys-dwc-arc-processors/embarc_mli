#
# Copyright 2019-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PUBLIC_DIR = $(CURDIR)/../..
include $(PUBLIC_DIR)/make/rules.mk

BIN_PATH = $(BUILD_DIR)$(PS)examples$(PS)example_face_detect$(PS)bin
RUN_ARGS ?=

ifndef TCF_FILE
# MSVC is build without Ninja, use MSVC/Makefile compatible settings
BUILD_SUBDIR = examples$(PS)example_face_detect
BUILD_TARGET = example_face_detect
else
# Ninja specific
BUILD_SUBDIR = ./
BUILD_TARGET = examples/example_face_detect/all
endif

ifndef TCF_FILE
RUN_APP_CMD =
else
RUN_APP_CMD = mdb64 -cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif

app: build

run: app
	$(RUN_APP_CMD) $(BIN_PATH)$(PS)$(BUILD_TARGET)$(BIN_EXT) $(RUN_ARGS)

.PHONY:	app run
