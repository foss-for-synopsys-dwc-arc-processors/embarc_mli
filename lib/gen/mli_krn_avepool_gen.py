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
func_body_template_file_chw = "mli_krn_avepool_chw_func_body.txt"
func_body_template_file_hwc = "mli_krn_avepool_hwc_func_body.txt"
file_template = "filetemplate.txt"
file_header_template = "header_filetemplate.txt"
function_group = "AvePooling"
capital_header_file_name = "_MLI_KRN_AVEPOOL_SPEC_API_H_"
output_header_file = "..\..\include\\api\mli_krn_avepool_spec_api.h"
output_file_chw_fx16 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_chw_fx16.cc"
output_file_chw_fx8 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_chw_fx8.cc"
output_file_hwc_fx16 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_hwc_fx16.cc"
output_file_hwc_fx8 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_hwc_fx8.cc"
output_file_hwc_sa8 = "..\..\lib\src\kernels\pooling\mli_krn_avepool_hwc_sa8.cc"

f_list_hwc = []
f_list_chw = []

f_args = [("const mli_tensor *", "in"),
          ("const mli_pool_cfg *", "cfg"),
          ("mli_tensor *", "out")]
fbase = ("krn", "avepool", "chw", "fx16", f_args)
include_list_chw = ["mli_krn_avepool_chw.h"]
include_list_hwc = ["mli_krn_avepool_hwc.h"]
define_list = []
# commandline arguments can be used to generate a specific output 'fx16' | 'fx8' | 'sa8' | 'header']
# if no arguments are given, all files are generated.
no_args = len(sys.argv) == 1

#------------------------------------------------------------
# Create a list of specialization functions for CHW
#------------------------------------------------------------
corefunc = "avepool_chw_krnpad"
stride = 0
kernel_range = range(2, 11)
ch = 0
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])

corefunc = "avepool_chw_nopad"
stride = 0
kernel_range = range(2, 11)
ch = 0
f_list_chw.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

corefunc = "avepool_chw_krnpad"
stride = 0
kernel_range = [0, 2, 3]
ch = 0
f_list_chw.extend([Func(fbase, 1, k, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])
f_list_chw.extend([Func(fbase, k, 1, ch, stride, stride, corefunc, "krnpad") for k in kernel_range])
corefunc = "avepool_chw_nopad"
f_list_chw.extend([Func(fbase, 1, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])
f_list_chw.extend([Func(fbase, k, 1, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

corefunc = "avepool_chw_krnpad"
default_func_chw = Func(fbase, 0, 0, 0, 0, 0, corefunc, generic=True)
f_list_chw.append(default_func_chw)

f_list_chw_fx16 = [f.copy_and_replace_base(fbase) for f in f_list_chw]
default_func_chw_fx16 = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Create a list of specialization functions for HWC
#------------------------------------------------------------
fbase = ("krn", "avepool", "hwc", "fx16", f_args)
corefunc = "avepool_hwc_krnpad"
stride = 0
k = 3
ch = 0
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "krnpad")])

corefunc = "avepool_hwc_nopad"
stride = 0
kernel_range = range(2, 11)
ch = 0
f_list_hwc.extend([Func(fbase, k, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

stride = 0
kernel_range = [2, 3]
ch = 0
corefunc = "avepool_hwc_nopad"
f_list_hwc.extend([Func(fbase, 1, k, ch, stride, stride, corefunc, "nopad") for k in kernel_range])
f_list_hwc.extend([Func(fbase, k, 1, ch, stride, stride, corefunc, "nopad") for k in kernel_range])

#at last add the generic function that can be used in the else branch in the wrapper.
corefunc = "avepool_hwc_krnpad"
default_func_hwc = Func(fbase, 0, 0, 0, 0, 0, corefunc, generic=True)
f_list_hwc.append(default_func_hwc)

f_list_hwc_fx16 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc_fx16 = default_func_hwc.copy_and_replace_base(fbase)
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

#------------------------------------------------------------
# Create a new list of specialization functions for CHW fx8
#------------------------------------------------------------
fbase = ("krn", "avepool", "chw", "fx8", f_args)

f_list_chw_fx8 = [f.copy_and_replace_base(fbase) for f in f_list_chw]
default_func_chw_fx8 = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Create a new list of specialization functions for HWC fx8
#------------------------------------------------------------
fbase = ("krn", "avepool", "hwc", "fx8", f_args)

f_list_hwc_fx8 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc_fx8 = default_func_chw.copy_and_replace_base(fbase)

#------------------------------------------------------------
# Create a new list of specialization functions for HWC sa8
#------------------------------------------------------------
fbase = ("krn", "avepool", "hwc", "sa8", f_args)

f_list_hwc_sa8 = [f.copy_and_replace_base(fbase) for f in f_list_hwc]
default_func_hwc_sa8 = default_func_chw.copy_and_replace_base(fbase)
#------------------------------------------------------------
# Generate the output file
#------------------------------------------------------------
if "fx16" in sys.argv or no_args:
    #Create FX16 CHW C output file
    f = open(output_file_chw_fx16, "wb")
    f.write(c.print_file(f_list_chw, default_func_chw_fx16, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()

    #Create FX16 HWC C output file
    f = open(output_file_hwc_fx16, "wb")
    f.write(c.print_file(f_list_hwc, default_func_hwc_fx16, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()
  

if "fx8" in sys.argv or no_args:
    #Create FX8 CHW C output file
    f = open(output_file_chw_fx8, "wb")
    f.write(c.print_file(f_list_chw_fx8, default_func_chw_fx8, func_body_template_file_chw, file_template, include_list_chw, define_list))
    f.close()
    
    #Create FX8 HWC C output file
    f = open(output_file_hwc_fx8, "wb")
    f.write(c.print_file(f_list_hwc_fx8, default_func_hwc_fx8, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()


if "sa8" in sys.argv or no_args:
    #Create SA8 HWC C output file
    f = open(output_file_hwc_sa8, "wb")
    f.write(c.print_file(f_list_hwc_sa8, default_func_hwc_sa8, func_body_template_file_hwc, file_template, include_list_hwc, define_list))
    f.close()

#------------------------------------------------------------
# Generate the output header file
#------------------------------------------------------------ 
if "header" in sys.argv or no_args:
    fh = open(output_header_file, "wb")
    fh.write(c.print_proto_file([f_list_chw_fx16,f_list_chw_fx8, f_list_hwc_fx16, f_list_hwc_fx8, f_list_hwc_sa8], function_group, capital_header_file_name, file_header_template))
    fh.close()

