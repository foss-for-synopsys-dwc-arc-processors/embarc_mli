#
# Copyright 2019, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# Default TCF is based on the standard EM9D Voice Audio template 
TCF ?= ../../hw/em9d.tcf

.PHONY: lib

all: app

lib:
	gmake -C lib/make TCF_FILE=$(TCF)

app: lib
	gmake -C lib/make TCF_FILE=$(TCF)
	gmake -C examples/example_cifar10_caffe TCF_FILE=$(TCF)
	gmake -C examples/example_har_smartphone TCF_FILE=$(TCF) 

cleanapp:
	gmake -C examples/example_cifar10_caffe clean 
	gmake -C examples/example_har_smartphone clean 

cleanall:
	gmake -C lib/make clean
	gmake -C examples/example_cifar10_caffe cleanall 
	gmake -C examples/example_har_smartphone cleanall

libclean:
	gmake -C lib/make clean