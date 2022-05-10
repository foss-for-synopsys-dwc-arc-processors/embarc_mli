#
# Copyright 2019-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

EMBARC_MLI_DIR := $(dir $(lastword $(MAKEFILE_LIST)))../..
include $(EMBARC_MLI_DIR)/make/rules.mk

BUILD_SUBDIR = examples$(PS)example_har_smartphone
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

ifndef TCF_FILE
RUN_APP_CMD =
else
RUN_APP_CMD = mdb64 -cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif

TYPES = \
	FX16 \
	SA8 \
	FX16_FX8_FX8

RUN_TARGETS = $(patsubst %,run_%,$(TYPES))

app: build

run: run_FX16

$(RUN_TARGETS): run_%: app
	$(RUN_APP_CMD) $(BIN_PATH)$(PS)example_har_smartphone_$*$(BIN_EXT) $(RUN_ARGS)

.PHONY:	app $(RUN_TARGETS)
