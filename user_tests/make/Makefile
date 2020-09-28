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
SRC_DIRS     = $(EMBARC_MLI_DIR)/user_tests/test_components \
               $(EMBARC_MLI_DIR)/user_tests/tests/mli_krn_softmax \
               $(EMBARC_MLI_DIR)/examples/auxiliary \


INC_DIRS     = $(EMBARC_MLI_DIR)/user_tests/test_components \
               $(EMBARC_MLI_DIR)/user_tests/tests/mli_krn_conv2d \
               $(EMBARC_MLI_DIR)/include \
               $(EMBARC_MLI_DIR)/examples/auxiliary

EXT_LIBS_DIR ?= $(EMBARC_MLI_DIR)/bin/arc_ref
EXT_LIBS     ?= $(EXT_LIBS_DIR)/libmli.a
OUT_DIR      ?= ./bin
BUILD_DIR    ?= ./obj
OUT_NAME     ?= test_mli_krn_conv2d
ifeq ($(TOOLCHAIN),mwdt)
# MWDT specific options
CFLAGS       =  -Hnocopyr -Hpurge -Hheap=16K -Hstack=16K -Hdsplib -Hfxapi -e_start -Bgrouplib -Hldopt=-q -Hsdata0 -Xdsp_ctrl=postshift,guard,convergent -Hdense_prologue

ifneq ($(RT_LIB),)
CFLAGS       +=  -Hlib=$(RT_LIB)
endif

else
PREBUILT_LIB ?= $(EMBARC_MLI_DIR)/examples/prebuilt/libmli.a

# GNU toolchain specific options - correct it according to your target platform settings (see build_configuration.txt for input)
#Iot DevKit config
CFLAGS       =  -mcpu=em4_fpuda -mlittle-endian -mcode-density -mdiv-rem -mswap -mnorm -mmpy-option=6 -mbarrel-shifter -mxy

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

.PHONY: clean all app
.DEFAULT_GOAL := all

all: app 

#$(EXT_LIBS): $(EXT_LIBS_DIR)
#	@echo Copy prebuilt library $(PREBUILT_LIB) to $@
#	@$(CP) $(call fix_platform_path,$(PREBUILT_LIB)) $(call fix_platform_path,$@)

#$(EXT_LIBS_DIR):
#	$(MKDIR) $(call fix_platform_path, $@ )

include $(EMBARC_MLI_DIR)/build/rules.mk

#ifeq ($(TOOLCHAIN),mwdt)
#lib:
#	@ $(MAKE) generic_lib -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS) TCF_FILE="$(TCF_FILE)"
#else
#lib: $(EXT_LIBS)
#endif

app: generic_app

run: generic_run

clean:
	@echo Cleaning application $(OUT_NAME)...
	-@$(RM) $(call fix_platform_path,$(OBJS))

cleanall: clean
	@echo Cleaning all files ..
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).elf)
	-@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).map)
#	+$(MAKE) clean -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS)
