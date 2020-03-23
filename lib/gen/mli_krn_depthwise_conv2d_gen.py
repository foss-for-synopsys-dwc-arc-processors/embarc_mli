from func import Func
from codegen import Codegen
import sys

# This script is used to generate the specialized versions for the depthwise_conv2d functions.
# The specialized functions can be called directly from the application, or the generated
# wrapper function can be called. The script builds a list with specializations by
# (optionally) fixing strides, kernel sizes, number of channels, or padding mode for
# different value ranges. A value 0 means that the specific parameter is not fixed.
# After the complete list is build, the code is generated based on a function template,
# and inserted into the file template. This script can be used to generate the cc files for
# different bit precisions, and it can also generate the header file that contains the
# function prototypes of all specializations. For normal operation of the lib there is no
# need to update the script.
# The script can be exectued with python 2.7

#------------------------------------------------------------
# convolution functions chw
#------------------------------------------------------------

#------------------------------------------------------------
# fill the basic information
#------------------------------------------------------------
func_body_template_file_chw = "mli_krn_depthwise_conv2d_chw_func_body.txt"
func_body_template_file_hwc = "mli_krn_depthwise_conv2d_hwcn_func_body.txt"
file_template = "filetemplate.txt"
file_header_template = "header_filetemplate.txt"
function_group = "Depthwise convolution 2d"
capital_header_file_name = "_MLI_KRN_DEPTHWISE_CONV2D_SPEC_API_H_"
output_header_file = "..\..\include\\api\mli_krn_depthwise_conv2d_spec_api.h"
output_file_chw_fx16 = "..\..\lib\src\kernels\convolution\mli_krn_depthwise_conv2d_chw_fx16.cc"
output_file_chw_fx8 = "..\..\lib\src\kernels\convolution\mli_krn_depthwise_conv2d_chw_fx8.cc"
output_file_chw_fx8w16d = "..\..\lib\src\kernels\convolution\mli_krn_depthwise_conv2d_chw_fx8w16d.cc"
output_file_hwcn_sa8 = "..\..\lib\src\kernels\convolution\mli_krn_depthwise_conv2d_hwcn_sa8_sa8_sa32.cc"

f_list_chw_fx16 = []
f_list_chw_fx8 = []
f_list_chw_fx8w16d = []
f_list_hwcn_sa8 = []

f_args = [("const mli_tensor *", "in"),
          ("const mli_tensor *", "weights"),
          ("const mli_tensor *", "bias"),
          ("const mli_conv2d_cfg *", "cfg"),
          ("mli_tensor *", "out")]
fbase = ("krn", "depthwise_conv2d", "chw", "fx16", f_args)
include_list_chw = ["mli_krn_conv2d_chw.h"]
include_list_hwc = ["mli_krn_conv2d_hwc.h"]
define_list = []
# commandline arguments can be used to generate a specific output 'fx16' | 'fx8' | 'header']
# if no arguments are given, all files are generated.
no_args = len(sys.argv) == 1


#------------------------------------------------------------
# Create a list of specialization functions for fx16
#------------------------------------------------------------

#stride = 1, fix dimension
corefunc = "conv2d_chw_str1"
stride = 1
f_list_chw_fx16.extend([Func(fbase, 1, 2, 1, stride, stride, corefunc, "krnpad")]) #k_width == 1
f_list_chw_fx16.extend([Func(fbase, 2, 1, 1, stride, stride, corefunc, "krnpad")]) #k_heigth == 1

#stride = 1, channels == 1
corefunc = "conv2d_chw_str1"
stride = 1
kernel_range = range(2,8)#range(1,6)
f_list_chw_fx16.extend([Func(fbase, k, k, 1, stride, stride, corefunc, "krnpad") for k in kernel_range ]) #channels == 1

#stride = 1, channels == any
corefunc = "conv2d_chw_str1"
stride = 1
kernel_range = range(2,8)#range(1,6)
channel_range = [0]#[0,1,3,4,8]
f_list_chw_fx16.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range for ch in channel_range])

#generic function for all the other stride=1 cases
corefunc = "conv2d_chw_str1"
stride = 1
k = 0
ch = 0
f_list_chw_fx16.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad")])

#stride = 1, fix dimension
corefunc = "conv2d_chw_str1"
stride = 1
f_list_chw_fx16.extend([Func(fbase, 1, 2, 1, stride, stride, corefunc, "nopad")]) #k_width == 1
f_list_chw_fx16.extend([Func(fbase, 2, 1, 1, stride, stride, corefunc, "nopad")]) #k_heigth == 1

#stride = 1, channels == 1
corefunc = "conv2d_chw_str1"
stride = 1
kernel_range = range(2,8)#range(1,6)
f_list_chw_fx16.extend([Func(fbase, k, k, 1, stride, stride, corefunc, "nopad") for k in kernel_range ]) #channels == 1

#stride = 1, channels == any
corefunc = "conv2d_chw_str1"
stride = 1
kernel_range = range(2,8)#range(1,6)
channel_range = [0]#[0,1,3,4,8]
f_list_chw_fx16.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range for ch in channel_range])

#fix single dimension, others flex
corefunc = "conv2d_chw_str1"
stride = 1
f_list_chw_fx16.extend([Func(fbase, 1, 0, 0, stride, stride, corefunc, "")]) #k_width == 1
f_list_chw_fx16.extend([Func(fbase, 0, 1, 0, stride, stride, corefunc, "")]) #k_heigth == 1
f_list_chw_fx16.extend([Func(fbase, 0, 0, 1, stride, stride, corefunc, "")]) #channels == 1


#at last add the generic function that can be used in the else branch in the wrapper.
default_func_chw = Func(fbase, 0, 0, 0, 0, 0, "conv2d_chw", generic=True)
f_list_chw_fx16.append(default_func_chw)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
c = Codegen()
c.set_wrapper_variables({'stride_w' : "cfg->stride_width", 'stride_h' : "cfg->stride_height"})
c.set_wrapper_variables({'kernel_w' : "weights->shape[KRNL_W_DIM_CHW]", 'kernel_h' : "weights->shape[KRNL_H_DIM_CHW]"})
c.set_wrapper_variables({'channels' : "in->shape[FMAP_C_DIM_CHW]"})
c.set_wrapper_variables({'padding_top' : "cfg->padding_top"})
c.set_wrapper_variables({'padding_bot' : "cfg->padding_bottom"})
c.set_wrapper_variables({'padding_left' : "cfg->padding_left"})
c.set_wrapper_variables({'padding_right' : "cfg->padding_right"})
c.set_wrapper_hierarchy(['stride_w', 'stride_h', 'kernel_w', 'kernel_h', 'channels', 'padding'])
c.set_wrapper_if_tree(False)

if "fx16" in sys.argv or no_args:
    f = open(output_file_chw_fx16, "wb")
    f.write(c.print_file(f_list_chw_fx16, default_func_chw, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for fx8
#------------------------------------------------------------


fbase = ("krn", "depthwise_conv2d", "chw", "fx8", f_args)

f_list_chw_fx8 = [f.copy_and_replace_base(fbase) for f in f_list_chw_fx16]
default_func_chw_fx8 = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx8" in sys.argv or no_args:
    f = open(output_file_chw_fx8, "wb")
    f.write(c.print_file(f_list_chw_fx8, default_func_chw_fx8, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for fx8w16d
#------------------------------------------------------------
fbase = ("krn", "depthwise_conv2d", "chw", "fx8w16d", f_args)

f_list_chw_fx8w16d = [f.copy_and_replace_base(fbase) for f in f_list_chw_fx8]
default_func_chw_fx8w16d = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx8w16d" in sys.argv or no_args:
    f = open(output_file_chw_fx8w16d, "wb")
    f.write(c.print_file(f_list_chw_fx8w16d, default_func_chw_fx8w16d, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

#------------------------------------------------------------
# Create a list of specialization functions for SA8 SA8 SA32
#------------------------------------------------------------

fbase = ("krn", "depthwise_conv2d", "hwcn", "sa8_sa8_sa32", f_args)

corefunc = "depthwise_convolution2D_hwcn_krnpad"
stride = 0
kernel_range = range(3, 6, 2)
ch = 0
f_list_hwcn_sa8.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])

corefunc = "depthwise_convolution2D_hwcn_nopad"
stride = 0
kernel_range = range(3, 6, 2)
ch = 0
f_list_hwcn_sa8.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

corefunc = "depthwise_convolution2D_hwcn_krnpad"
default_func_hwc = Func(fbase, 0, 0, 0, 0, 0, corefunc, generic=True)
f_list_hwcn_sa8.append(default_func_hwc)

#------------------------------------------------------------
# Generate the HWC output file
#------------------------------------------------------------
c = Codegen()
c.set_wrapper_variables({'kernel_w' : "weights->shape[KRNL_DW_W_DIM_HWC]", 'kernel_h' : "weights->shape[KRNL_DW_H_DIM_HWC]"})
c.set_wrapper_variables({'padding_top' : "cfg->padding_top"})
c.set_wrapper_variables({'padding_bot' : "cfg->padding_bottom"})
c.set_wrapper_variables({'padding_left' : "cfg->padding_left"})
c.set_wrapper_variables({'padding_right' : "cfg->padding_right"})
c.set_wrapper_hierarchy(['kernel_w', 'kernel_h', 'padding'])
c.set_wrapper_if_tree(False)

if "sa8_sa8_sa32" in sys.argv or no_args:
    #Create SA8 HWC C output file
    f = open(output_file_hwcn_sa8, "wb")
    f.write(c.print_file(f_list_hwcn_sa8, default_func_hwc, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()


#------------------------------------------------------------
# Generate the output header file
#------------------------------------------------------------
if "header" in sys.argv or no_args:
    fh = open(output_header_file, "wb")
    fh.write(c.print_proto_file([f_list_chw_fx16,f_list_chw_fx8,f_list_chw_fx8w16d, f_list_hwcn_sa8], function_group, capital_header_file_name, file_header_template))
    fh.close()

