#
# Copyright 2019-2022, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

EMBARC_MLI_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(EMBARC_MLI_DIR)/lib/make/makefile
