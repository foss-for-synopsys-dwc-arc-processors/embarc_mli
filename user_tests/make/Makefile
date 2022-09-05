#
# Copyright 2020-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

EMBARC_MLI_DIR := $(dir $(lastword $(MAKEFILE_LIST)))/../../
include $(EMBARC_MLI_DIR)/make/rules.mk

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
	tensor_struct\

KERNELS = \
	permute \
	conv2d \
	depthwise_conv \
	group_conv2d_FX16 \
	group_conv2d_FX16_FX8_FX8 \
	group_conv2d_SA8_SA8_SA32 \
	transpose_conv2d \
	fully_connected \
	rnn_dense \
	lstm_cell_FX16 \
	lstm_cell_FX16_FX8_FX8 \
	lstm_cell_SA8_SA8_SA32 \
	gru_cell_FX16 \
	gru_cell_FX16_FX8_FX8 \
	gru_cell_SA8_SA8_SA32 \
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
	l2_normalize \
	data_movement \
	data_movement_30 \
	maxpool_30 \
	sumpool_30 \
	fully_connected_30 \
	eltwise_30 \
	conv2d_30 \
	depthwise_conv_30 \
	reduce_max_30 \
	transpose_conv2d_30



BIN_FILES=$(patsubst %,$(BIN_PATH)$(PS)test_mli_hlp_%$(BIN_EXT),$(HELPERS))
BIN_FILES+=$(patsubst %,$(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT),$(KERNELS))
TEST_KRN_TARGETS = $(patsubst %,test_%,$(KERNELS))
TEST_HLP_TARGETS = $(patsubst %,test_%,$(HELPERS))

# Rules (see rules.mk for more rules)
#===========================================
$(BIN_FILES): build

test_all: $(TEST_HLP_TARGETS) $(TEST_KRN_TARGETS)

$(TEST_KRN_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<

$(TEST_HLP_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_hlp_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<

get_tests_list:
	@echo $(TEST_HLP_TARGETS) $(TEST_KRN_TARGETS)

.PHONY: test_all $(TEST_KRN_TARGETS) $(TEST_HLP_TARGETS) get_tests_list
