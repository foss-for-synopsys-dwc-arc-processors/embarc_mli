#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PLATFORMLIST = \
    emsdp_em11d_em9d_dfss \
    emsdp_em7d_em5d_dfss \
    himax_arcem9d_r16 \
    iotdk_arcem9d


LIB_DIR = ../../bin
LIB_NAME = libmli.a
TCF_DIR = ../../hw
LIB_LIST = $(addsuffix /$(LIB_NAME), $(addprefix $(LIB_DIR)/, $(PLATFORMLIST)) )
PKG_NAME = embARC_MLI_package
include ../../build/rules.mk 
PKG_TMP_FOLDER = ../../$(PKG_NAME)
FILE_LIST = $(addsuffix .tcf, $(addprefix $(PKG_TMP_FOLDER)/hw/, $(PLATFORMLIST)) )
FILE_LIST += $(PKG_TMP_FOLDER)/LICENSE

$(LIB_LIST) : $(LIB_DIR)/%/$(LIB_NAME): $(TCF_DIR)/%.tcf
	$(MAKE) TCF_FILE=$< BUILD_DIR=../../obj/$*/debug LIBRARY_DIR=$(LIB_DIR)/$*/debug DEBUG_BUILD=ON EXT_CFLAGS="-DMLI_DEBUG_MODE=DBG_MODE_FULL"
	$(MAKE) TCF_FILE=$< BUILD_DIR=../../obj/$*/release LIBRARY_DIR=$(LIB_DIR)/$*/release DEBUG_BUILD=OFF EXT_CFLAGS="-DMLI_DEBUG_MODE=DBG_MODE_RELEASE"

$(PKG_TMP_FOLDER):
	$(MKDIR) $(call fix_platform_path,$@)
	$(MKDIR) $(call fix_platform_path,$@/hw)

package_libs: $(LIB_LIST) $(PKG_TMP_FOLDER)
	$(CPR) $(call fix_platform_path,../../bin) $(call fix_platform_path,$(PKG_TMP_FOLDER)/bin/)

package_includes: $(PKG_TMP_FOLDER)
	$(CPR) $(call fix_platform_path,../../include) $(call fix_platform_path,../../$(PKG_NAME)/include/)

$(FILE_LIST): $(PKG_TMP_FOLDER)/% : ../../%
	$(CP) $(call fix_platform_path,$<) $(call fix_platform_path,$@)

package_content: package_libs package_includes $(FILE_LIST)

package: package_content
	cd ../.. & zip $(PKG_NAME).zip -r $(PKG_NAME)
	$(RMDIR) $(call fix_platform_path,$(PKG_TMP_FOLDER))

clean:
	@echo Cleaning package...
	-@$(RMDIR) $(call fix_platform_path,$(LIB_DIR))
	-@$(RMDIR) $(call fix_platform_path,../../obj)
	-@$(RM) $(call fix_platform_path,../../$(PKG_NAME).zip)
