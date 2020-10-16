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
VERBOSE ?= false
PUBLIC_DIR = ..$(PS)..
BUILD_SUBDIR = user_tests
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin
# User need to define BUILD_LIB (target specific libs including runtime)
# if default one doesn't fit.
BUILDLIB_DIR ?=

ifndef TCF_FILE

RUN_TEST_CMD =
ifeq ($(O_SYS),Windows)
BIN_EXT = .exe
else 
BIN_EXT = 
endif

else
RUN_TEST_CMD = mdb -run  -tcf=$(TCF_FILE)
BIN_EXT = .elf
endif


BIN_FILES = \
	$(BIN_PATH)$(PS)test_mli_krn_softmax$(BIN_EXT) \
	$(BIN_PATH)$(PS)test_mli_krn_eltwise$(BIN_EXT) \
	$(BIN_PATH)$(PS)test_mli_krn_conv2d$(BIN_EXT) \
	$(BIN_PATH)$(PS)test_mli_krn_avepool$(BIN_EXT) \
	$(BIN_PATH)$(PS)test_mli_krn_maxpool$(BIN_EXT)

TEST_TARGETS = test_softmax test_conv2d test_eltwise test_avepool test_maxpool



# Build rules to generate project, library and test apps
#===========================================
$(BUILD_DIR): 
	$(MAKE) -C $(PUBLIC_DIR)$(PS)lib$(PS)make RECONFIGURE=ON config

$(BIN_FILES): build


# Public rules to run tests or clean current test build
#====================================================
build: $(BUILD_DIR)
	cmake --build $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) --verbose


clean:
# Check that cmake generated a Makefile project. 
# In this case we have an extra assumption to make a more tidy clean 
# rule (remove only application objects/binaries).  Otherwise, it may not work 
# for several project types (as MSVC) and requires cleanin all solution
ifneq ($(realpath $(BUILD_DIR)$(PS)Makefile),)
	cmake --build $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) --target clean
else
	cmake --build $(BUILD_DIR) --target clean
endif


cleanall:
	$(MAKE) -C $(PUBLIC_DIR)$(PS)lib$(PS)make clean


test_all: $(TEST_TARGETS)


test_conv2d: $(BIN_PATH)$(PS)test_mli_krn_conv2d$(BIN_EXT)
	$(RUN_TEST_CMD) $<


test_avepool: $(BIN_PATH)$(PS)test_mli_krn_avepool$(BIN_EXT)
	$(RUN_TEST_CMD) $<


test_maxpool: $(BIN_PATH)$(PS)test_mli_krn_maxpool$(BIN_EXT)
	$(RUN_TEST_CMD) $<


test_softmax: $(BIN_PATH)$(PS)test_mli_krn_softmax$(BIN_EXT)
	$(RUN_TEST_CMD) $<


test_eltwise: $(BIN_PATH)$(PS)test_mli_krn_eltwise$(BIN_EXT)
	$(RUN_TEST_CMD) $<


.PHONY: build clean cleanall test_all $(TEST_TARGETS)
