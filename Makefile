#
# Copyright 2019-2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

# Default TCF is based on the standard EM9D Voice Audio template 
TCF_FILE ?= ../../hw/em9d.tcf

.PHONY: lib

all: app

lib:
	$(MAKE) -C lib/make TCF_FILE=$(TCF_FILE)

app: lib
	$(MAKE) -C lib/make TCF_FILE=$(TCF_FILE)
	$(MAKE) -C examples/example_cifar10_caffe TCF_FILE=$(TCF_FILE)
	$(MAKE) -C examples/example_har_smartphone TCF_FILE=$(TCF_FILE) 
	$(MAKE) -C examples/example_face_detect TCF_FILE=$(TCF_FILE)

cleanapp:
	$(MAKE) -C examples/example_cifar10_caffe clean 
	$(MAKE) -C examples/example_har_smartphone clean 
	$(MAKE) -C examples/example_face_detect clean

cleanall:
	$(MAKE) -C lib/make clean
	$(MAKE) -C examples/example_cifar10_caffe cleanall 
	$(MAKE) -C examples/example_har_smartphone cleanall
	$(MAKE) -C examples/example_face_detect cleanall

libclean:
	$(MAKE) -C lib/make clean

package:
	$(MAKE) -C lib/make -f package.mk package

cleanpackage:
	$(MAKE) -C lib/make -f package.mk clean
