#
# Copyright 2019, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# Target toolchain. The library project doesn't support GCC
TOOLCHAIN = mwdt
TCF_FILE ?= ../../hw/em9d.tcf

# Directories and files
SRC_ROOT    =  ../src
SRC_DIRS    =  ../src/helpers/src \
               ../src/kernels/common \
               ../src/kernels/convolution \
               ../src/kernels/data_manip \
               ../src/kernels/eltwise \
               ../src/kernels/pooling \
               ../src/kernels/transform \
               ../src/private/src

INC_DIRS    = .  \
              ../../include \
              ../../include/api \
              ../src/private \
              ../src/helpers \
              ../src/kernels \
              ../src/kernels/eltwise \
              ../src/kernels/pooling \
              ../src/kernels/transform

# source files that are generated
CONVOLUTION_SRC_FOLDER = ../src/kernels/convolution
POOLING_SRC_FOLDER = ../src/kernels/pooling
GEN_HEADER_FOLDER = ../../include/api

GEN_FILES_BASE = $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d \
                 $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d \
                 $(POOLING_SRC_FOLDER)/mli_krn_avepool \
                 $(POOLING_SRC_FOLDER)/mli_krn_maxpool

GEN_FILES  = $(addsuffix _chw_fx16.cc, $(GEN_FILES_BASE))
GEN_FILES += $(addsuffix _chw_fx8.cc, $(GEN_FILES_BASE))
GEN_FILES += $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_fx8w16d.cc \
             $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_fx8w16d.cc

GEN_FILES_HEADERS = $(addprefix $(GEN_HEADER_FOLDER)/, $(addsuffix _spec_api.h, $(notdir $(GEN_FILES_BASE))))

EXT_LIBS     = 
LIBRARY_DIR ?= ../../bin
OUT_NAME    ?= libmli
BUILD_DIR   ?= ../../obj
GEN_DIR     ?= ../gen
CFLAGS      += -O3 -Xdsp_ctrl=postshift,guard,convergent -Hnocopyr -Hfxapi -Hpurge -Hsdata0

ifneq ($(ComSpec)$(COMSPEC),)
    O_SYS=Windows
    DEV_NULL = nul
else
    O_SYS=Unix
    DEV_NULL=/dev/null
endif

# Variable with all default flags and defines for target HW (tcf file).
ALL_FLAGS := $(strip $(shell ccac -tcf=$(TCF_FILE) -Hbatchnotmp _.c 2> $(DEV_NULL)))
# Analize all flags.
ifeq (-av2hs,$(findstring -av2hs, $(ALL_FLAGS)))
#add extra cflag for HS platforms
CFLAGS += -Hpipeline
else
#add extra cflag for EM platforms
CFLAGS += -Hdense_prologue
endif

vpath %.py  $(GEN_DIR)
vpath %.txt  $(GEN_DIR)
vpath %.h  $(INC_DIRS)

all: lib

include ../../build/rules.mk


.PHONY: clean

#lib: scripts generic_lib
lib: generic_lib

run:
	$(error Unsupported target 'run')

clean:
	@echo Cleaning library...
	-@$(RM) $(call fix_platform_path,$(OBJS))
	-@$(RM) $(call fix_platform_path,$(LIBRARY_DIR)/$(OUT_NAME).a)


$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%.cc : mli_krn_conv2d_gen.py mli_krn_conv2d_func_body.txt filetemplate.txt
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%.cc : mli_krn_depthwise_conv2d_gen.py mli_krn_depthwise_conv2d_func_body.txt filetemplate.txt
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%.cc : mli_krn_avepool_gen.py mli_krn_avepool_func_body.txt filetemplate.txt
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_maxpool_chw_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_maxpool_chw_%.cc : mli_krn_maxpool_gen.py mli_krn_maxpool_func_body.txt filetemplate.txt
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(GEN_FILES_HEADERS): $(GEN_HEADER_FOLDER)/%_spec_api.h : %_gen.py header_filetemplate.txt
	@echo [generate] $@
	cd $(GEN_DIR) && python $< header


gen: $(GEN_FILES) $(GEN_FILES_HEADERS)
