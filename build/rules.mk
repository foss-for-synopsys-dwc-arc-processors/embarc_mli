#
# Copyright 2019, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

#=============================================================
# OS-specific definitions
#=============================================================
COMMA=,
OPEN_PAREN=(
CLOSE_PAREN=)
BACKSLASH=\$(nullstring)
ifneq ($(ComSpec)$(COMSPEC),)
    O_SYS=Windows
    RM=del /F /Q /S
    MKDIR=mkdir 
    CP=copy /Y
    TYPE=type
    PS=$(BACKSLASH)
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
    Q=$(BACKSLASH)
    coQ=
    fix_platform_path=$(1)
    DEV_NULL=/dev/null
endif

# Note: Windows escaping rules is very combersome 
# initially I tried to use Q=^, but this depends on the context and (looks like) on Win version.
# Also expecially ugly thing is that in quoted strings the quotes the same are remain.
# Batch has special parameter expansion syntax to remove quotes,
# but many tools themselves remove quotes (unless escaped with backslash)
# So finally we've found that in our use cases we may not escaping any symbols but prepend backslashes before quotes.

quote=$(subst %,$(Q)%, \
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
      $(subst ",$(BACKSLASH)", \
      $(subst $(Q),$(Q)$(Q), \
      $(1) )))))))))))))


#=============================================================
# Global settings
#=============================================================
TOOLCHAIN ?= gnu
#optmization mode
OPTMODE ?= speed

export DEBUG_BUILD?=ON
#export ASM_OUT?=OFF

ifeq ($(DEBUG_BUILD),ON)
    CFLAGS += -g
endif
#ifeq ($(ASM_OUT),ON)
#    CFLAGS += -g0 -Hkeepasm -Hanno
#    # CFLAGS += -Hon=Print_var_info
#endif

ifeq ($(OPTMODE),size)
	CFLAGS += -O2 -Hlto
endif
ifeq ($(OPTMODE),speed)
	CFLAGS += -O3
endif

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
 TCF_CFLAGS += -tcf=$(TCF_FILE) -tcf_core_config
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

# TCF file needs to be the in front of the other CFLAGS
CFLAGS  := $(TCF_CFLAGS) $(call quote,  $(CFLAGS))
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
ifneq ($(CC_DEPENDS),)
-include $(CC_DEPENDS)
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

