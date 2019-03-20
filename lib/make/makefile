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

EXT_LIBS     = 
LIBRARY_DIR ?= ../../bin
OUT_NAME    ?= libmli
BUILD_DIR   ?= ../../obj
CFLAGS      += -O3 -g -Xdsp_ctrl=postshift,guard,convergent -Hnocopyr -Hfxapi -Hpurge -Hsdata0

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

