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

#=============================================================
# OS-specific definitions
#=============================================================
ifneq ($(ComSpec)$(COMSPEC),)
    O_SYS=Windows
    RM=del /F /Q
    MKDIR=mkdir 
    CP=copy /Y
    TYPE=type
    PS=\$(nullstring)
    Q=
    coQ=\$(nullstring)
    fix_platform_path = $(subst /,$(PS), $(1))
    DEV_NULL = nul
else
    O_SYS=Unix
    RM=rm -rf
    MKDIR=mkdir -p
    CP=cp 
    TYPE=cat
    PS=/
    Q=\$(nullstring)
    coQ=^
    fix_platform_path=$(1)
    DEV_NULL=/dev/null
endif

COMMA=,
OPEN_PAREN=(
CLOSE_PAREN=)
quote=$(subst $(coQ),$(Q)$(coQ), \
      $(subst %,$(Q)%, \
      $(subst &,$(Q)&, \
      $(subst <,$(Q)<, \
      $(subst >,$(Q)>, \
      $(subst |,$(Q)|, \
      $(subst ',$(Q)', \
      $(subst $(COMMA),$(Q)$(COMMA), \
      $(subst =,$(Q)=, \
      $(subst $(OPEN_PAREN),$(Q)$(OPEN_PAREN), \
      $(subst $(CLOSE_PAREN),$(Q)$(CLOSE_PAREN), \
      $(subst !,$(Q)!, \
      $(subst ",$(Q)", \
      $(subst $(Q),$(Q)$(Q), \
	  $(1) ))))))))))))))


#=============================================================
# Global settings
#=============================================================
TOOLCHAIN ?= gnu

export DEBUG_BUILD?=ON
#export ASM_OUT?=OFF

ifeq ($(DEBUG_BUILD),ON)
    CFLAGS += -g
endif
#ifeq ($(ASM_OUT),ON)
#    CFLAGS += -g0 -Hkeepasm -Hanno
#    # CFLAGS += -Hon=Print_var_info
#endif

#=============================================================
# Files and directories
#=============================================================


# Sources and objects lists
C_SRCS 		= $(basename $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c))))
CPP_SRCS 	= $(basename $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))))
CC_SRCS 	= $(basename $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cc))))

C_OBJS		= $(addsuffix .o, $(addprefix $(BUILD_DIR)/, $(C_SRCS)) )
CPP_OBJS	= $(addsuffix .o, $(addprefix $(BUILD_DIR)/, $(CPP_SRCS)) )
CC_OBJS		= $(addsuffix .o, $(addprefix $(BUILD_DIR)/, $(CC_SRCS)) )

C_DEPNDS		= $(addsuffix .d, $(addprefix $(BUILD_DIR)/, $(C_SRCS)) )
CPP_DEPENDS	= $(addsuffix .d, $(addprefix $(BUILD_DIR)/, $(CPP_SRCS)) )
CC_DEPENDS	= $(addsuffix .d, $(addprefix $(BUILD_DIR)/, $(CC_SRCS)) )

OBJS		= $(C_OBJS) $(CPP_OBJS) $(CC_OBJS) $(C_DEPNDS) $(CPP_DEPENDS) $(CC_DEPENDS)
# 
# Toolchain-specific parts
#
ifeq ($(TOOLCHAIN),gnu)
# place to add GNU-specific common settings
CFLAGS     +=-fno-short-enums
DEPFLAGS    =
else
ifeq ($(TOOLCHAIN),mwdt)
# place to add MWDT-specific common settings
CFLAGS     +=-Hon=Long_enums
DEPFLAGS    =-Hdepend=$(BUILD_DIR)/ -MD
else
$(error ERROR - Unsupported toolchain. Supported toolchains are 'gnu' and 'mwdt', default one is 'gnu')
endif
endif

CFLAGS += $(addprefix -I, $(INC_DIRS))
CFLAGS += $(EXT_CFLAGS)

#=============================================================
# Toolchain definitions
#=============================================================

# Tools
ifeq ($(TOOLCHAIN),mwdt)
 CC = ccac
 LD = ccac
 AR = arac
 AS = ccac
 CFLAGS += -tcf=$(TCF_FILE) -tcf_core_config
 LDFLAGS += -tcf=$(TCF_FILE) $(LCF)
else 
 CC = arc-elf32-gcc
 LD = arc-elf32-ld
 AR = arc-elf32-ar
 AS = arc-elf32-as
 CFLAGS += $(addprefix -I, $(HEADER_DIRS))
 CFLAGS += -D_Interrupt=__attribute__((interrupt("ilink")))
 CFLAGS += -D_lr=__builtin_arc_lr
 CFLAGS += -D_sr=__builtin_arc_sr
 CFLAGS += -D_seti=__builtin_arc_seti
 CFLAGS += -D_clri=__builtin_arc_clri
 CFLAGS += -D_kflag=__builtin_arc_kflag
 #CFLAGS += -D_sleep=__builtin_arc_sleep
 CFLAGS += -D__Xdmac

 CFLAGS += -D_Uncached=volatile
 CFLAGS += -D_Usually(x)=__builtin_expect((x)!=0,1)
 CFLAGS += -D_Rarely(x)=__builtin_expect((x)!=0,0)
 CFLAGS += -DIRQ_BUILD=0x00f3
 CFLAGS += -DRF_BUILD=0x006e
 CFLAGS += -DRF_BUILD=0x006e
 CFLAGS += -DSTATUS32=0x000a
 LDFLAGS += -marcv2elfx
 LDFLAGS += -lc
 LDFLAGS += -lm 
 LDFLAGS += -lnsim
 LDFLAGS += --gc-sections 
 ifneq ($(LCF),) 
  LDFLAGS +=--script=$(LCF)
 endif
endif

CFLAGS  := $(call quote,  $(CFLAGS))
LDFLAGS := $(call quote, $(LDFLAGS))

vpath %.c  $(SRC_DIRS) 
vpath %.cpp  $(SRC_DIRS) 
vpath %.cc  $(SRC_DIRS)
vpath %.o  $(BUILD_DIR)
vpath %.d  $(BUILD_DIR)


#=============================================================
# Common rules
#=============================================================
$(C_OBJS): $(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
ifeq ($(BUILD_DIR),)
	$(error cannot build C object file $@ -- BUILD_DIR variable must be set)
endif
	@echo [CC] $<
	$(CC) $(CFLAGS) $(INCS) -c $(DEPFLAGS) $< -o $@

$(CPP_OBJS): $(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
ifeq ($(BUILD_DIR),)
	$(error cannot build C++ object file $@ -- BUILD_DIR variable must be set)
endif
	@echo [CPP] $<
	$(CC) $(CFLAGS) $(INCS) -c $(DEPFLAGS) $< -o $@

$(CC_OBJS): $(BUILD_DIR)/%.o: %.cc | $(BUILD_DIR)
ifeq ($(BUILD_DIR),)
	$(error cannot build C++ object file $@ -- BUILD_DIR variable must be set)
endif
	@echo [C++] $<
	$(CC) $(CFLAGS) $(INCS) -c $(DEPFLAGS) $< -o $@

$(LIBRARY_DIR):
	$(MKDIR) $(call fix_platform_path, $(LIBRARY_DIR))

$(BUILD_DIR):
	$(MKDIR) $(call fix_platform_path, $(BUILD_DIR))

$(OUT_DIR):
	$(MKDIR) $(call fix_platform_path, $(OUT_DIR))

#=============================================================
# Library rules
#=============================================================
generic_lib: $(LIBRARY_DIR)/$(OUT_NAME).a

$(LIBRARY_DIR)/$(OUT_NAME).a: $(C_OBJS) $(CPP_OBJS) $(CC_OBJS) | $(LIBRARY_DIR)
ifeq ($(LIBRARY_DIR),)
	$(error cannot create library file $@ -- LIBRARY_DIR variable must be set)
endif
ifeq ($(OUT_NAME),)
	$(error cannot create library file $@ -- OUT_NAME variable must be set)
endif
	@echo Archiving $@...
	@$(AR) -cq $@ $(C_OBJS) $(CPP_OBJS) $(CC_OBJS)

ifneq ($(C_DEPNDS),)
-include $(C_DEPNDS)
endif
ifneq ($(CPP_DEPENDS),)
-include $(CPP_DEPENDS)
endif

#=================================================================
# Applications rules
#=================================================================
generic_app: $(OUT_DIR)/$(OUT_NAME).elf

$(OUT_DIR)/$(OUT_NAME).elf: $(C_OBJS) $(CPP_OBJS) $(CC_OBJS) | $(OUT_DIR)
ifeq ($(OUT_DIR),)
	$(error cannot create executable $@ -- OUT_DIR variable must be set)
endif
ifeq ($(OUT_NAME),)
	$(error cannot create executable $@ -- OUT_NAME variable must be set)
endif
	@echo Linking $@...
ifeq ($(TOOLCHAIN),mwdt)
	$(CC) $(CFLAGS) $(C_OBJS) $(CPP_OBJS) $(CC_OBJS) $(EXT_LIBS) $(addprefix -Wl$(COMMA),$(LDFLAGS)) -o $@ -m > $(OUT_DIR)/$(OUT_NAME).map
else
	$(CC) $(CFLAGS) $(C_OBJS) $(CPP_OBJS) $(CC_OBJS) $(EXT_LIBS) $(addprefix -Wl$(COMMA),$(LDFLAGS)) --specs=nsim.specs -o $@
endif


#=================================================================
# Execution rules
#=================================================================

ifeq ($(RUN_METHOD),FPGA_RUN)
ELF_RUN=mdb $(MDB_ARGS) -digilent -nohostlink_while_running -nooptions -noproject
endif

RUN_METHOD?=NSIM_RUN

MDB_ARGS?=-run -cl

ifeq ($(RUN_METHOD),NSIM_RUN)
ELF_RUN=mdb $(MDB_ARGS) -nsim -tcf=$(TCF_FILE) -profile $(DBG_OPTS)
endif

ifeq ($(RUN_METHOD),NSIM_DEBUG)
ELF_RUN=mdb -nsim -tcf=$(TCF_FILE) $(DBG_OPTS)
endif

ifeq ($(RUN_METHOD),XCAM_RUN)
ELF_RUN=mdb $(MDB_ARGS) -nsim -tcf=$(TCF_FILE) -profile $(DBG_OPTS) 
endif

ifeq ($(RUN_METHOD),VCS_RUN)
ELF_RUN=mdb -cl -OK -off=cr_for_more -rascal -run -nogoifmain -prop=direct_mem_link=download  
endif

ifeq ($(TOOLCHAIN),gnu)
ELF_RUN+=-prop=nsim_emt=1
endif


generic_run: $(OUT_DIR)/$(OUT_NAME).elf
	$(ELF_RUN) $(OUT_DIR)/$(OUT_NAME).elf $(RUN_ARGS)

.PHONY: generic_lib generic_app generic_run packet_run

#show: 
#	@echo -e "$(foreach v, $(.VARIABLES), $(origin $(v)) $(v)=$($(v))\n )"
#.PHONY: show

