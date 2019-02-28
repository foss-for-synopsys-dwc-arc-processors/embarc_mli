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

# default toolchain
TOOLCHAIN ?= mwdt
TCF_FILE ?= ../../hw/em9d.tcf

# Directories and files
PUBLIC_DIR ?= ../..
SRC_DIRS 	= $(PUBLIC_DIR)/examples/example_har_smartphone \
			  $(PUBLIC_DIR)/examples/auxiliary

INC_DIRS 	= . \
			  $(PUBLIC_DIR)/include\
			  $(PUBLIC_DIR)/examples/auxiliary


EXT_LIBS	?= $(PUBLIC_DIR)/bin/libmli.a
OUT_DIR		?= ./bin
BUILD_DIR	?= ./obj
OUT_NAME	?= example_har_smartphone
CFLAGS		=  -Hnocopyr -Hpurge -Hheap=16K -Hstack=4K -Hfxapi -e_start -Bgrouplib -Hldopt=-q -O0

all: lib app 

include $(PUBLIC_DIR)/build/rules.mk

.PHONY: clean all lib cleanall app
.DEFAULT: all

lib:
	@ $(MAKE) generic_lib -C ..$(PS)..$(PS)lib$(PS)make$(PS) TCF_FILE="$(TCF_FILE)"

app: generic_app

run: generic_run

clean:
	@echo Cleaning application $(OUT_NAME)...
	-@$(RM) $(call fix_platform_path,$(OBJS))

cleanall: clean
	@echo Cleaning all files ..
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).elf)
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).map)
	+$(MAKE) clean -C ..$(PS)..$(PS)lib$(PS)make$(PS)
