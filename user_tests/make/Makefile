#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include ../../lib/make/settings.mk

# Configuring variables
#===========================================
PUBLIC_DIR = ..$(PS)..
BUILD_SUBDIR = user_tests
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin

ifndef TCF_FILE
RUN_TEST_CMD =
else
RUN_TEST_CMD = mdb -run -jit  -tcf=$(TCF_FILE)
endif


BIN_FILES = \
	$(BIN_PATH)$(PS)test_mli_krn_softmax.elf \
	$(BIN_PATH)$(PS)test_mli_krn_conv2d.elf
	
TEST_TARGETS = test_softmax test_conv2d 



# Build rules to generate project, library and test apps
#===========================================
$(BUILD_DIR): 
	$(MAKE) -C $(PUBLIC_DIR)$(PS)lib$(PS)make lib

$(BIN_FILES): $(BUILD_DIR)
	$(MAKE) -C $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) VERBOSE=true install



# Public rules to run tests or clean current test build
#====================================================
build: $(BIN_FILES)

test_all: $(TEST_TARGETS)


test_conv2d: $(BIN_PATH)$(PS)test_mli_krn_conv2d.elf
	$(RUN_TEST_CMD) $<


test_softmax: $(BIN_PATH)$(PS)test_mli_krn_softmax.elf
	$(RUN_TEST_CMD) $<


clean:
	$(MAKE) -C $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) VERBOSE=true clean


cleanall:
	$(MAKE) -C $(PUBLIC_DIR)$(PS)lib$(PS)make clean
	

.PHONY: build clean cleanall test_all $(TEST_TARGETS)
