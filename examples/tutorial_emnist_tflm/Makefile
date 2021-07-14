#
# Copyright 2019-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# TCF_FILE: withouts NOT setting this parameter,
#           the default native compiler is used to compile
# BUILDLIB_DIR ?= ev71_buildlib can be specified if no MWDT buildlib is available
# See rules.mk for all flags

PUBLIC_DIR = $(CURDIR)/../..

ifeq ($(TENSORFLOW_DIR),)
    $(error cannot find tensorflow -- TENSORFLOW_DIR variable must be set)
endif

BUILD_SUBDIR = examples$(PS)tutorial_emnist_tflm
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

.PHONY: app

include $(PUBLIC_DIR)/make/rules.mk

ifndef TCF_FILE
	RUN_APP_CMD =
else
	RUN_APP_CMD = mdb -cl -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
	BIN_EXT = .elf
endif

app: build

run: app
	$(RUN_APP_CMD) $(BIN_PATH)$(PS)tutorial_emnist_tflm$(BIN_EXT) $(RUN_ARGS)