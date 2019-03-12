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

# default hardware config file
TCF_FILE  ?= ../../hw/em9d.tcf

# Directories and files
EMBARC_MLI_DIR ?= ../..
SRC_DIRS     = $(EMBARC_MLI_DIR)/examples/example_cifar10_caffe \
               $(EMBARC_MLI_DIR)/examples/auxiliary

INC_DIRS     = . \
               $(EMBARC_MLI_DIR)/include \
               $(EMBARC_MLI_DIR)/examples/auxiliary

EXT_LIBS_DIR ?= $(EMBARC_MLI_DIR)/bin
EXT_LIBS     ?= $(EXT_LIBS_DIR)/libmli.a
OUT_DIR      ?= ./bin
BUILD_DIR    ?= ./obj
OUT_NAME     ?= example_cifar10_caffe
ifeq ($(TOOLCHAIN),mwdt)
# MWDT specific options
CFLAGS       =  -Hnocopyr -Hpurge -Hheap=8K -Hstack=1K -Hfxapi -e_start -Bgrouplib -Hldopt=-q -O0
else
PREBUILT_LIB ?= $(EMBARC_MLI_DIR)/examples/prebuilt/libmli.a

# GNU toolchain specific options - correct it according to your target platform settings (see build_configuration.txt for input)
#Iot DevKit config
CFLAGS       =  -mcpu=em4_dmips -mlittle-endian -mcode-density -mdiv-rem -mswap -mnorm -mmpy-option=6 -mbarrel-shifter -mxy

# The embARC MLI Library specific options it according to your target platform settings 
#(EM5D or EM7D platform)
#CFLAGS      += -DV2DSP
#(EM9D or EM11D platform)
CFLAGS      += -DV2DSP_XY
#(HS45D or HS47D platform)
#CFLAGS      += -DV2DSP_WIDE

# GNU toolchain linker specific options
LDFLAGS      = --defsym=__DEFAULT_HEAP_SIZE=8k 
LDFLAGS     += --defsym=__DEFAULT_STACK_SIZE=1k
LDFLAGS     += -Map $(OUT_DIR)/$(OUT_NAME).map

#specific options for run the example with the MetaWare Debuger on the nSim simulator. 
DBG_OPTS     = -cmd="read mdb_com_gnu"
endif

.PHONY: clean all lib cleanall app
.DEFAULT_GOAL := all

all: lib app 

$(EXT_LIBS): $(EXT_LIBS_DIR)
	@echo Copy prebuilt library $(PREBUILT_LIB) to $@
	@$(CP) $(call fix_platform_path,$(PREBUILT_LIB)) $(call fix_platform_path,$@)

$(EXT_LIBS_DIR):
	$(MKDIR) $(call fix_platform_path, $@ )

include $(EMBARC_MLI_DIR)/build/rules.mk

ifeq ($(TOOLCHAIN),mwdt)
lib:
	@ $(MAKE) generic_lib -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS) TCF_FILE="$(TCF_FILE)"
else
lib: $(EXT_LIBS)
endif

app: generic_app

run: generic_run

clean:
	@echo Cleaning application $(OUT_NAME)...
	-@$(RM) $(call fix_platform_path,$(OBJS))

cleanall: clean
	@echo Cleaning all files ..
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).elf)
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).map)
	+$(MAKE) clean -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS)
