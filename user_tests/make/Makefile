#
# Copyright 2020-2021, Synopsys, Inc.
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
RUN_TEST_CMD = mdb64 -run -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif

HELPERS = \
	convert_tensor\

KERNELS = \
	permute \
	conv2d \
	depthwise_conv \
	group_conv2d \
	transpose_conv2d \
	fully_connected \
	rnn_dense \
	avepool \
	maxpool \
	argmax \
	softmax \
	relu \
	leaky_relu \
	prelu \
	eltwise \
	tanh \
	sigm \
	l2_normalize\
	data_movement


BIN_FILES=$(patsubst %,$(BIN_PATH)$(PS)test_mli_hlp_%$(BIN_EXT),$(HELPERS))
BIN_FILES+=$(patsubst %,$(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT),$(KERNELS))
TEST_KRN_TARGETS = $(patsubst %,test_%,$(KERNELS)) 
TEST_HLP_TARGETS = $(patsubst %,test_%,$(HELPERS)) 

# Rules (see rules.mk for more rules)
#===========================================
$(BIN_FILES): build

test_all: $(TEST_HLP_TARGETS) $(TEST_TARGETS)

$(TEST_KRN_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<
    
$(TEST_HLP_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_hlp_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<

get_tests_list:
	@echo $(TEST_HLP_TARGETS) $(TEST_KRN_TARGETS)

.PHONY: test_all $(TEST_KRN_TARGETS) $(TEST_HLP_TARGETS) get_tests_list
