from func import Func
from codegen import Codegen
import sys

# This script is used to generate the specialized versions for the maxpool functions.
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
# maxpool functions chw
#------------------------------------------------------------

#------------------------------------------------------------
# fill the basic information
#------------------------------------------------------------
func_body_template_file_chw = "mli_krn_maxpool_chw_func_body.txt"
func_body_template_file_hwc = "mli_krn_maxpool_hwc_func_body.txt"
file_template = "filetemplate.txt"
file_header_template = "header_filetemplate.txt"
function_group = "MaxPooling"
capital_header_file_name = "_MLI_KRN_MAXPOOL_SPEC_API_H_"
output_header_file = "..\..\include\\api\mli_krn_maxpool_spec_api.h"
output_file_chw_fx16 = "..\..\lib\src\kernels\pooling\mli_krn_maxpool_chw_fx16.cc"
output_file_chw_fx8 = "..\..\lib\src\kernels\pooling\mli_krn_maxpool_chw_fx8.cc"
output_file_hwc_fx8 = "..\..\lib\src\kernels\pooling\mli_krn_maxpool_hwc_fx8.cc"
output_file_hwc_fx16 = "..\..\lib\src\kernels\pooling\mli_krn_maxpool_hwc_fx16.cc"
output_file_hwc_int8 = "..\..\lib\src\kernels\pooling\mli_krn_maxpool_hwc_int8.cc"

f_list_chw = []
f_list_hwc = []
f_args = [("const mli_tensor *", "in"),
          ("const mli_pool_cfg *", "cfg"),
          ("mli_tensor *", "out")]
fbase = ("krn", "maxpool", "chw", "fx16", f_args)
include_list_chw = ["mli_krn_maxpool_chw.h"]
include_list_hwc = ["mli_krn_maxpool_hwc.h"]
define_list = []
# commandline arguments can be used to generate a specific output 'fx16' | 'fx8' | 'header']
# if no arguments are given, all files are generated.
no_args = len(sys.argv) == 1

#------------------------------------------------------------
# Create a list of specialization functions for fx16
#------------------------------------------------------------

corefunc_chw = "maxpool_chw_nopad"
stride = 1
kernel_range = [2]
channel_range = [0]
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc_chw, "nopad") for k in kernel_range for ch in channel_range])

corefunc_chw = "maxpool_chw_nopad"
stride = 0
kernel_range_chw = range(2,11)
channel_range = [0]
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc_chw, "nopad") for k in kernel_range_chw for ch in channel_range])

#stride = 0, 1xk and kx1 versions
corefunc_chw = "maxpool_chw_nopad"
stride = 0
kernel_range = range(2,4)
channel_range = [0]
f_list_chw.extend([Func(fbase, 1, k, ch, stride, stride, corefunc_chw, "nopad") for k in kernel_range for ch in channel_range])
f_list_chw.extend([Func(fbase, k, 1, ch, stride, stride, corefunc_chw, "nopad") for k in kernel_range for ch in channel_range])

corefunc_chw = "maxpool_chw_krnpad_small"
stride = 0
kernel_range = [2, 3]
channel_range = [0]
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc_chw, "krnpad") for k in kernel_range for ch in channel_range])

corefunc_chw = "maxpool_chw_pad"
stride = 0
kernel_range_chw = range(4,11)
channel_range = [0]
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc_chw, "krnpad") for k in kernel_range_chw for ch in channel_range])

# 1xk and kx1 versions
corefunc_chw = "maxpool_chw_pad"
stride = 0
kernel_range = range(2,4)
channel_range = [0]
f_list_chw.extend([Func(fbase, 1, k, ch, stride, stride, corefunc_chw, "krnpad") for k in kernel_range for ch in channel_range])
f_list_chw.extend([Func(fbase, k, 1, ch, stride, stride, corefunc_chw, "krnpad") for k in kernel_range for ch in channel_range])

#fix single dimension, others flex
corefunc_chw = "maxpool_chw_pad"
stride = 0
f_list_chw.extend([Func(fbase, 1, 0, 0, stride, stride, corefunc_chw, "")]) #k_width == 1
f_list_chw.extend([Func(fbase, 0, 1, 0, stride, stride, corefunc_chw, "")]) #k_heigth == 1

corefunc_chw = "maxpool_chw_pad"
stride = 0
kernel_range = [2,3]
channel_range = [0]
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc_chw, "") for k in kernel_range for ch in channel_range])

#at last add the generic function that can be used in the else branch in the wrapper.
corefunc_chw = "maxpool_chw_pad"
default_func_chw = Func(fbase, 0, 0, 0, 0, 0, corefunc_chw, generic=True)
f_list_chw.append(default_func_chw)

corefunc_hwc = "maxpool_hwc_nopad"
stride = 1
kernel_range = [2]
channel_range = [0]
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc_hwc, "nopad") for k in kernel_range for ch in channel_range])

corefunc_hwc = "maxpool_hwc_nopad"
stride = 0
kernel_range_hwc = range(2, 8)
channel_range = [0]
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc_hwc, "nopad") for k in kernel_range_hwc for ch in channel_range])

#stride = 0, 1xk and kx1 versions
corefunc_hwc = "maxpool_hwc_nopad"
stride = 0
kernel_range = range(2,4)
channel_range = [0]
f_list_hwc.extend([Func(fbase, 1, k, ch, stride, stride, corefunc_hwc, "nopad") for k in kernel_range for ch in channel_range])
f_list_hwc.extend([Func(fbase, k, 1, ch, stride, stride, corefunc_hwc, "nopad") for k in kernel_range for ch in channel_range])

corefunc_hwc = "maxpool_hwc_pad"
stride = 0
kernel_range_hwc = range(4,6)
channel_range = [0]
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc_hwc, "krnpad") for k in kernel_range_hwc for ch in channel_range])

# 1xk and kx1 versions
corefunc_hwc = "maxpool_hwc_pad"
stride = 0
kernel_range = range(2,4)
channel_range = [0]
f_list_hwc.extend([Func(fbase, 1, k, ch, stride, stride, corefunc_hwc, "krnpad") for k in kernel_range for ch in channel_range])
f_list_hwc.extend([Func(fbase, k, 1, ch, stride, stride, corefunc_hwc, "krnpad") for k in kernel_range for ch in channel_range])

corefunc_hwc = "maxpool_hwc_pad"
stride = 0
kernel_range = [2,3]
channel_range = [0]
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc_hwc, "") for k in kernel_range for ch in channel_range])

#at last add the generic function that can be used in the else branch in the wrapper.
corefunc_hwc = "maxpool_hwc_pad"
default_func_hwc = Func(fbase, 0, 0, 0, 0, 0, corefunc_hwc, generic=True)
f_list_hwc.append(default_func_hwc)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
c = Codegen()
c.set_wrapper_variables({'stride_w' : "cfg->stride_width", 'stride_h' : "cfg->stride_height"})
c.set_wrapper_variables({'kernel_w' : "cfg->kernel_width", 'kernel_h' : "cfg->kernel_height"})
c.set_wrapper_variables({'padding_top' : "cfg->padding_top"})
c.set_wrapper_variables({'padding_bot' : "cfg->padding_bottom"})
c.set_wrapper_variables({'padding_left' : "cfg->padding_left"})
c.set_wrapper_variables({'padding_right' : "cfg->padding_right"})
c.set_wrapper_hierarchy(['stride_w', 'stride_h', 'kernel_w', 'kernel_h', 'padding'])
c.set_wrapper_if_tree(False)

if "fx16" in sys.argv or no_args:
    f = open(output_file_chw_fx16, "wb")
    f.write(c.print_file(f_list_chw, default_func_chw, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for fx8
#------------------------------------------------------------

fbase = ("krn", "maxpool", "chw", "fx8", f_args)

f_list_chw_fx8 = [f.copy_and_replace_base(fbase) for f in f_list_chw]
default_func_chw = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx8" in sys.argv or no_args:
    f = open(output_file_chw_fx8, "wb")
    f.write(c.print_file(f_list_chw_fx8, default_func_chw, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for hwc fx8
#------------------------------------------------------------

fbase = ("krn", "maxpool", "hwc", "fx8", f_args)

f_list_hwc_fx8 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc = default_func_hwc.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx8" in sys.argv or no_args:
    f = open(output_file_hwc_fx8, "wb")
    f.write(c.print_file(f_list_hwc_fx8, default_func_hwc, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for hwc fx16
#------------------------------------------------------------

fbase = ("krn", "maxpool", "hwc", "fx16", f_args)

f_list_hwc_fx16 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc = default_func_hwc.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx16" in sys.argv or no_args:
    f = open(output_file_hwc_fx16, "wb")
    f.write(c.print_file(f_list_hwc_fx16, default_func_hwc, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for hwc int8
#------------------------------------------------------------

fbase = ("krn", "maxpool", "hwc", "int8", f_args)

f_list_hwc_int8 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc = default_func_hwc.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "int8" in sys.argv or no_args:
    f = open(output_file_hwc_int8, "wb")
    f.write(c.print_file(f_list_hwc_int8, default_func_hwc, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()

#------------------------------------------------------------
# Generate the output header file
#------------------------------------------------------------
if "header" in sys.argv or no_args:
    fh = open(output_header_file, "wb")
    fh.write(c.print_proto_file([f_list_chw, f_list_chw_fx8, f_list_hwc_fx8, f_list_hwc_fx16, f_list_hwc_int8], function_group, capital_header_file_name, file_header_template))
    fh.close()
