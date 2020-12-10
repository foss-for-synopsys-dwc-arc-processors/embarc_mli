#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PUBLIC_DIR ?= $(CURDIR)/../../
include $(PUBLIC_DIR)/make/rules.mk

# Configuring variables (see rules.mk for more variables like BUILDLIB_DIR)
#==========================================================================
BUILD_SUBDIR = user_tests
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

ifndef TCF_FILE
RUN_TEST_CMD =
else
RUN_TEST_CMD = mdb -run  -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif

KERNELS = \
	hlp_convert_tensor\
	conv2d \
	depthwise_conv \
	group_conv2d \
	transpose_conv2d \
	fully_connected \
	avepool \
	maxpool \
	softmax \
	relu \
	leaky_relu \
	prelu \
	eltwise \
	tanh \
	sigm \
	l2_normalize

BIN_FILES=$(patsubst %,$(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT),$(KERNELS))
TEST_TARGETS = $(patsubst %,test_%,$(KERNELS))

# Rules (see rules.mk for more rules)
#===========================================
$(BIN_FILES): build

test_all: $(TEST_TARGETS)

$(TEST_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<

get_tests_list:
	@echo $(TEST_TARGETS)

.PHONY: test_all $(TEST_TARGETS) get_tests_list
