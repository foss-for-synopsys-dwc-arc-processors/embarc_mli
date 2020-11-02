#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PUBLIC_DIR ?= ../..
include $(PUBLIC_DIR)/lib/make/settings.mk

# Configuring variables
#===========================================
VERBOSE ?= false
BUILD_SUBDIR = user_tests
BIN_PATH = $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)bin
# User need to define BUILD_LIB (target specific libs including runtime)
# if default one doesn't fit.
BUILDLIB_DIR ?=
PERFORM_BUILD?=ON

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

KERNELS=conv2d depthwise_conv fully_connected avepool maxpool softmax relu eltwise

BIN_FILES=$(patsubst %,$(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT),$(KERNELS))
TEST_TARGETS = $(patsubst %,test_%,$(KERNELS))



# Build rules to generate project, library and test apps
#===========================================
$(BUILD_DIR):
	$(MAKE) -C $(PUBLIC_DIR)$(PS)lib$(PS)make RECONFIGURE=ON config

$(BIN_FILES): build


# Public rules to run tests or clean current test build
#====================================================
build: $(BUILD_DIR)
ifeq ($(PERFORM_BUILD),ON)
	cmake --build $(BUILD_DIR)$(PS)$(BUILD_SUBDIR) --verbose
endif

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

$(TEST_TARGETS): test_%: $(BIN_PATH)$(PS)test_mli_krn_%$(BIN_EXT)
	$(RUN_TEST_CMD) $<

get_tests_list:
	@echo $(TEST_TARGETS)

.PHONY: build clean cleanall test_all $(TEST_TARGETS)
