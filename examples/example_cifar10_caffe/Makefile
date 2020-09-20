#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# TCF_FILE: withouts NOT setting this parameter,
#           the default native compiler is used to compile
# BUILDLIB_DIR ?= ev71_buildlib can be specified if no MWDT buildlib is available

include ../../lib/make/settings.mk
PUBLIC_DIR = ..$(PS)..$(PS)
BUILD_SUBDIR = examples$(PS)example_cifar10_caffe

# Targets
app:
	$(MAKE) -C $(PUBLIC_DIR)lib$(PS)make lib BUILD_SUBDIR=$(BUILD_SUBDIR)

clean:
	$(MAKE) -C $(PUBLIC_DIR)lib$(PS)make clean

ifndef TCF_FILE
run: app
	$(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)elf_files$(PS)example_cifar10_caffe
else
run: app
	mdb $(BUILD_DIR)$(PS)$(BUILD_SUBDIR)$(PS)elf_files$(PS)example_cifar10_caffe.elf \
		-cl  -nsim -cmd=run -off=cr_for_more -cmd=exit -tcf=$(TCF_FILE)
endif

.PHONY:	app clean run
