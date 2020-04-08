#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

PLATFORMLIST = em9d

LIB_DIR = ../../bin
LIB_NAME = libmli.a
TCF_DIR = ../../hw
LIB_LIST = $(addsuffix /$(LIB_NAME), $(addprefix $(LIB_DIR)/, $(PLATFORMLIST)) ) 

include ../../build/rules.mk 


$(LIB_LIST) : $(LIB_DIR)/%/$(LIB_NAME): $(TCF_DIR)/%.tcf
	gmake TCF_FILE=$< BUILD_DIR=../../obj/$* LIBRARY_DIR=$(LIB_DIR)/$*


package_content: $(LIB_LIST)

package: package_content
	cd ../../ & zip package.zip -r include bin & cd lib/make

clean:
	@echo Cleaning package...
	-@$(RM) $(call fix_platform_path,$(LIB_DIR))
	-@$(RM) $(call fix_platform_path,../../obj)
	-@$(RM) $(call fix_platform_path,../../package.zip)
