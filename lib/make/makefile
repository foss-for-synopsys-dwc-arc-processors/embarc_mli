# Copyright (c) 2019, Synopsys, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1) Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2)  Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3) Neither the name of the <ORGANIZATION> nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

