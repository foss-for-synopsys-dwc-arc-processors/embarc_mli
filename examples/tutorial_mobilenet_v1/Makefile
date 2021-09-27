#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# default toolchain
TOOLCHAIN ?= mwdt

# default hardware config file
TCF_FILE  ?= ../../hw/em9d.tcf

# Directories and files
EMBARC_MLI_DIR ?= ../..

ifeq ($(TENSORFLOW_DIR),)
    $(error cannot find tensorflow -- TENSORFLOW_DIR variable must be set)
endif

SRC_DIRS     = $(EMBARC_MLI_DIR)/examples/tutorial_mobilenet_v1/src

INC_DIRS     = ./src \
               $(EMBARC_MLI_DIR)/include \
               $(EMBARC_MLI_DIR)/include/api \
			   $(TENSORFLOW_DIR) \
			   $(TENSORFLOW_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
			   $(TENSORFLOW_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include 

EXT_LIBS_DIR ?= $(EMBARC_MLI_DIR)/bin
EXT_LIBS     ?= ./third_party/libtensorflow-microlite.a $(EXT_LIBS_DIR)/libmli.a 
OUT_DIR      ?= ./bin
BUILD_DIR    ?= ./obj
OUT_NAME     ?= example_mobilenet_v1

CFLAGS = -DTF_LITE_STATIC_MEMORY -tcf_core_config -Hnocopyr -Hpurge -Hdense_prologue -Hon=Long_enums -fslp-vectorize-aggressive -ffunction-sections -fdata-sections -Hcl -Hcrt_fast_memcpy -Hcrt_fast_memset -Hheap=80K -Hstack=16K 

DBG_OPTS = -jit 

.PHONY: clean all lib cleanall app
#.DEFAULT_GOAL := all

include $(EMBARC_MLI_DIR)/build/rules.mk

ELF_RUN := $(filter-out -profile,$(ELF_RUN))

all: lib app 

lib:
	@ $(MAKE) generic_lib -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS) TCF_FILE="$(TCF_FILE)"

app: src/model.h generic_app

run: generic_run

clean:
	@echo Cleaning application $(OUT_NAME)...
	-@$(RM) $(call fix_platform_path,$(OBJS))    
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).elf)
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).map)

cleanall: clean
	@echo Cleaning all files ..
	@ $(MAKE) clean -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS)