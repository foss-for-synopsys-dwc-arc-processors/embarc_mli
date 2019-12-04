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

# Make rules for code generation are default disabled to avoid
# dependency to python when re-generation is not needed
GEN_RULES ?= OFF

# Directories and files
SRC_ROOT    =  ../src
SRC_DIRS    =  ../src/helpers/src \
               ../src/kernels/common \
               ../src/kernels/convolution \
               ../src/kernels/data_manip \
               ../src/kernels/eltwise \
               ../src/kernels/pooling \
               ../src/kernels/transform \
               ../src/private/src \
               ../src/move

INC_DIRS    = .  \
              ../../include \
              ../../include/api \
              ../src/private \
              ../src/helpers \
              ../src/kernels \
              ../src/kernels/eltwise \
              ../src/kernels/pooling \
              ../src/kernels/transform \
              ../src/move

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
             $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_fx8w16d.cc \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_fx8.cc \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_fx16.cc \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_sa8.cc

GEN_FILES_HEADERS = $(addprefix $(GEN_HEADER_FOLDER)/, $(addsuffix _spec_api.h, $(notdir $(GEN_FILES_BASE))))

EXT_LIBS     = 
LIBRARY_DIR ?= ../../bin
OUT_NAME    ?= libmli
BUILD_DIR   ?= ../../obj
GEN_DIR     ?= ../gen
CFLAGS      += -Xdsp_ctrl=postshift,guard,convergent -Hnocopyr -Hfxapi -Hpurge -Hsdata0 -Hdense_prologue

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


ifeq ($(GEN_RULES),ON)

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%.cc : mli_krn_conv2d_gen.py mli_krn_conv2d_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%.cc : mli_krn_depthwise_conv2d_gen.py mli_krn_depthwise_conv2d_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%.cc : mli_krn_avepool_gen.py mli_krn_avepool_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_maxpool_chw_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_maxpool_chw_%.cc : mli_krn_maxpool_gen.py mli_krn_maxpool_chw_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_%.cc : mli_krn_maxpool_gen.py mli_krn_maxpool_hwc_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(GEN_FILES_HEADERS): $(GEN_HEADER_FOLDER)/%_spec_api.h : %_gen.py header_filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< header

gen: $(GEN_FILES) $(GEN_FILES_HEADERS)

endif
