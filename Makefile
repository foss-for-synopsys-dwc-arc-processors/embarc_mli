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

# Default TCF is EM7D voice audio provided as a part of MWDT toolchain
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