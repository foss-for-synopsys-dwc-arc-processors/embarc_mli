#
# Copyright 2019-2020, Synopsys, Inc.
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

# Collect iformation about platform from TCF
ALL_PLATFORM_FLAGS := '$(shell ccac -tcf=$(TCF_FILE) -Hbatchnotmp _.c)'
# Define type of platform
ifneq (,$(filter +vdsp, $(ALL_PLATFORM_FLAGS)))
MLI_PLATFORM = VPX
else
MLI_PLATFORM = EM_HS
endif

ifeq ($(MLI_PLATFORM),VPX)
SRC_DIRS    =  ../src/helpers/src \
               ../src/kernels/eltwise
else
SRC_DIRS    =  ../src/helpers/src \
               ../src/kernels/common \
               ../src/kernels/convolution \
               ../src/kernels/data_manip \
               ../src/kernels/eltwise \
               ../src/kernels/pooling \
               ../src/kernels/pooling_chw \
               ../src/kernels/transform \
               ../src/private/src \
               ../src/move
endif

INC_DIRS    = .  \
              ../../include \
              ../../include/api \
              ../src/private \
              ../src/helpers \
              ../src/kernels \
              ../src/kernels/eltwise \
              ../src/kernels/pooling \
              ../src/kernels/pooling_chw \
              ../src/kernels/transform \
              ../src/move \
              ../src/pal

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
             $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_hwc_sa8_sa8_sa32.cc \
             $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_fx8w16d.cc \
             $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_hwc_sa8_sa8_sa32.cc \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_fx8.cc  \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_fx16.cc \
             $(POOLING_SRC_FOLDER)/mli_krn_avepool_hwc_sa8.cc  \
             $(POOLING_SRC_FOLDER)/mli_krn_maxpool_hwc_sa8.cc

GEN_FILES_HEADERS = $(addprefix $(GEN_HEADER_FOLDER)/, $(addsuffix _spec_api.h, $(notdir $(GEN_FILES_BASE))))

EXT_LIBS     = 
LIBRARY_DIR ?= ../../bin
OUT_NAME    ?= libmli
BUILD_DIR   ?= ../../obj
GEN_DIR     ?= ../gen
CFLAGS      += -Hnocopyr -Hpurge -Hsdata0 -Hdense_prologue -Wall

ifeq ($(MLI_PLATFORM),VPX)
    CFLAGS  += -Hvdsp_vector_c -mllvm -slot_swapping=true
else
ifeq ($(MLI_PLATFORM),EM_HS)
    CFLAGS  += -Xdsp_ctrl=postshift,guard,convergent -Hfxapi
else
    $(error invalid platform)
endif
endif
ifeq ($(MLI_BUILD_REFERENCE),ON)
    CFLAGS += -DMLI_BUILD_REFERENCE
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


ifeq ($(GEN_RULES),ON)

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_chw_%.cc : mli_krn_conv2d_gen.py mli_krn_conv2d_chw_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_hwc_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_conv2d_hwc_%.cc : mli_krn_conv2d_gen.py mli_krn_conv2d_hwc_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_chw_%.cc : mli_krn_depthwise_conv2d_gen.py mli_krn_depthwise_conv2d_chw_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_hwc_%, $(GEN_FILES)): $(CONVOLUTION_SRC_FOLDER)/mli_krn_depthwise_conv2d_hwc_%.cc : mli_krn_depthwise_conv2d_gen.py mli_krn_depthwise_conv2d_hwc_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_avepool_chw_%.cc : mli_krn_avepool_gen.py mli_krn_avepool_chw_func_body.txt filetemplate.txt codegen.py func.py
	@echo [generate] $@
	cd $(GEN_DIR) && python $< $*

$(filter $(POOLING_SRC_FOLDER)/mli_krn_avepool_hwc_%, $(GEN_FILES)): $(POOLING_SRC_FOLDER)/mli_krn_avepool_hwc_%.cc : mli_krn_avepool_gen.py mli_krn_avepool_hwc_func_body.txt filetemplate.txt codegen.py func.py
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
