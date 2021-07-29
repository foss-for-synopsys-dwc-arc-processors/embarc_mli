#
# Copyright 2019-2021, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PUBLIC_DIR = $(CURDIR)/../..
include $(PUBLIC_DIR)/make/rules.mk

ifeq ($(TENSORFLOW_DIR),)
    $(error cannot find tensorflow -- TENSORFLOW_DIR variable must be set)
endif

BUILD_SUBDIR = examples$(PS)tutorial_emnist_tflm
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

ifndef TCF_FILE
	RUN_APP_CMD =
else
	RUN_APP_CMD = mdb -cl -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
	BIN_EXT = .elf
endif

export TCF_FILE_NAME = $(basename $(notdir $(TCF_FILE)))
app: build

run:
	$(RUN_APP_CMD) $(BIN_PATH)$(PS)tutorial_emnist_tflm$(BIN_EXT) $(RUN_ARGS)

.PHONY: app