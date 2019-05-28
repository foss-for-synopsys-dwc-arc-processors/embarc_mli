from func import Func
from codegen import Codegen
import sys

# This script is used to generate the specialized versions for the avepool functions.
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
# avepool functions chw
#------------------------------------------------------------

#------------------------------------------------------------
# fill the basic information
#------------------------------------------------------------
func_body_template_file = "mli_krn_avepool_func_body.txt"
file_template = "filetemplate.txt"
file_header_template = "header_filetemplate.txt"
function_group = "AvePooling"
capital_header_file_name = "_MLI_KRN_AVEPOOL_SPEC_API_H_"
output_header_file = "..\..\include\\api\mli_krn_avepool_spec_api.h"
output_file_fx16 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_chw_fx16.cc"
output_file_fx8 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_chw_fx8.cc"

f_list = []
f_args = [("const mli_tensor *", "in"),
          ("const mli_pool_cfg *", "cfg"),
          ("mli_tensor *", "out")]
fbase = ("krn", "avepool", "chw", "fx16", f_args)
include_list = ["mli_krn_avepool_chw.h"]
define_list = []
# commandline arguments can be used to generate a specific output 'fx16' | 'fx8' | 'header']
# if no arguments are given, all files are generated.
no_args = len(sys.argv) == 1

#------------------------------------------------------------
# Create a list of specialization functions
#------------------------------------------------------------

#construct the different specializtions for stride 1 and kernel 2x2
corefunc = "avepool_chw_nopad_k2x2"
stride = 1
k = 2
ch = 0
f_list.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad")])

corefunc = "avepool_chw_k4x4_str1_nopad"
stride = 1
k = 4
ch = 0
f_list.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad")])

#stride = 1, any kernel size, any channel size
corefunc = "avepool_chw_krnpad"
stride = 1
kernel_range = range(5, 11, 2)
ch = 0
f_list.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])

corefunc = "avepool_chw_krnpad_k4_Nx2_N_even"
stride = 1
width_range = range(4, 9, 2)
height_range = range(2, 9, 2)
ch = 0
f_list.extend([Func(fbase, w, h, ch, stride, stride, corefunc, "krnpad") for w in width_range for h in height_range])

corefunc = "avepool_chw_nopad_k4_Nx2_N_even"
stride = 1
w = 4
h = 2
ch = 0
f_list.extend([Func(fbase, w, h, ch, stride, stride, corefunc, "nopad")])

#stride = 1, any kernel size, any channel size
corefunc = "avepool_chw_nopad_k4_Nx2_N_even"
stride = 1
width_range = range(6, 9, 2)
height_range = range(2, 9, 2)
ch = 0
f_list.extend([Func(fbase, w, h, ch, stride, stride, corefunc, "nopad") for w in width_range for h in height_range])

#stride = 1, any kernel size, any channel size
corefunc = "avepool_chw_nopad"
stride = 1
kernel_range = range(3, 11, 2)
ch = 0
f_list.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

#here construct the specializations for any stride, and multiple kernel sizes > 1
corefunc = "avepool_chw_krnpad"
stride = 0
kernel_range = range(2, 11)
ch = 0
f_list.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])

#at last add the generic function that can be used in the else branch in the wrapper.
corefunc = "avepool_chw_krnpad"
default_func = Func(fbase, 0, 0, 0, 0, 0, corefunc, generic=True)
f_list.append(default_func)

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
c.set_wrapper_if_tree(True)

if "fx16" in sys.argv or no_args:
    f = open(output_file_fx16, "wb")
    f.write(c.print_file(f_list, default_func, func_body_template_file, file_template, include_list, define_list))
    f.close()

#------------------------------------------------------------
# Create a new list of specialization functions for fx8
#------------------------------------------------------------
fbase = ("krn", "avepool", "chw", "fx8", f_args)

f_list_fx8 = [f.copy_and_replace_base(fbase) for f in f_list]
default_func = default_func.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx8" in sys.argv or no_args:
    f = open(output_file_fx8, "wb")
    f.write(c.print_file(f_list_fx8, default_func, func_body_template_file, file_template, include_list, define_list))
    f.close()

#------------------------------------------------------------
# Generate the output header file
#------------------------------------------------------------
if "header" in sys.argv or no_args:
    fh = open(output_header_file, "wb")
    fh.write(c.print_proto_file([f_list,f_list_fx8], function_group, capital_header_file_name, file_header_template))
    fh.close()

