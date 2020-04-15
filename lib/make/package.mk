#
# Copyright 2020, Synopsys, Inc.
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

include ../../build/rules.mk 


$(LIB_LIST) : $(LIB_DIR)/%/$(LIB_NAME): $(TCF_DIR)/%.tcf
	$(MAKE) TCF_FILE=$< BUILD_DIR=../../obj/$*/debug LIBRARY_DIR=$(LIB_DIR)/$*/debug DEBUG_BUILD=ON EXT_CFLAGS="-DMLI_DEBUG_MODE=DBG_MODE_FULL"
	$(MAKE) TCF_FILE=$< BUILD_DIR=../../obj/$*/release LIBRARY_DIR=$(LIB_DIR)/$*/release DEBUG_BUILD=OFF EXT_CFLAGS="-DMLI_DEBUG_MODE=DBG_MODE_RELEASE"


package_content: $(LIB_LIST)

package: package_content
	cd ../../ & zip package.zip -r include bin & cd lib/make

clean:
	@echo Cleaning package...
	-@$(RM) $(call fix_platform_path,$(LIB_DIR))
	-@$(RM) $(call fix_platform_path,../../obj)
	-@$(RM) $(call fix_platform_path,../../package.zip)
