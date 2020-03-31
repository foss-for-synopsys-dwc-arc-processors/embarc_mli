from string import Template

class Func:
    kernel_w = 0
    kernel_h  = 0
    channels = 0
    group = ""
    typen = ""
    layout = ""
    datatype = ""
    padding = ""
    args = []
    inaccuracy = 7
    max_len_of_line = 120 + inaccuracy

    def __init__(self, base, kernel_w, kernel_h, channels, stride_w, stride_h, corefunc, padding="", generic=False):
        self.group, self.typen, self.layout, self.datatype, self.args = base
        
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.channels = channels
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.padding = padding
        if isinstance(corefunc, list):
            self.corefunc = corefunc
        else:
            self.corefunc = [corefunc]
        self.generic = generic
        self.debug = False

    def get_param_by_name(self, paramname):
        if paramname == 'kernel_w':
            return self.kernel_w
        if paramname == 'kernel_h':
            return self.kernel_h
        if paramname == 'stride_w':
            return self.stride_w
        if paramname == 'stride_h':
            return self.stride_h
        if paramname == 'channels':
            return self.channels
        if paramname == 'padding':
            return self.padding

    def get_base(self):
        return (self.group, self.typen, self.layout, self.datatype, self.args)

    def get_base_func(self):
        return Func(self.get_base(), 0, 0, 0, 0, 0, "")

    def copy_and_replace_base(self, base):
        return Func(base, self.kernel_w, self.kernel_h, self.channels,
                    self.stride_w, self.stride_h, self.corefunc, self.padding, self.generic)

    def name(self):
        name = "mli"
        if self.debug:
            name += "_debug"
        name += "_{0}_{1}_{2}_{3}".format(self.group, self.typen, self.layout, self.datatype)
        if ((self.kernel_w > 0) and (self.kernel_h > 0)):
            name += "_k{0}x{1}".format(self.kernel_w, self.kernel_h)
        if ((self.kernel_w > 0) and (self.kernel_h == 0)):
            name += "_k{0}xn".format(self.kernel_w)
        if ((self.kernel_w == 0) and (self.kernel_h > 0)):
            name += "_knx{0}".format(self.kernel_h)
        if (self.channels > 0):
            name += "_ch{0}".format(self.channels)
        if (self.stride_w > 0) and (self.stride_w == self.stride_h):
            name += "_str{0}".format(self.stride_w)
        if (self.stride_w > 0) and (self.stride_w != self.stride_h):
            name += "_strw{0}".format(self.stride_w)
        if (self.stride_h > 0) and (self.stride_w != self.stride_h):
            name += "_strh{0}".format(self.stride_h)
        if self.padding:
            name += "_"+ self.padding
        if self.generic:
            name += "_generic"
        return name

    def argstr(self, withtype=True, newline="", split=False):
        #put "\n" in newline to put every argument at new line
        string = "("
        sep = newline
        for a in self.args:
            dtype, argname = a
            string += sep
            if split:
                string += "\n" + newline + " " * 8
            if withtype:
                string += dtype + " "
            string += argname
            sep = ", " + newline
        string += ")"
        return string

    def print_proto(self, returntype = "mli_status"):
        func_len = len(returntype + " " + self.name() + self.argstr(True))
        if func_len > self.max_len_of_line:
            return returntype + " " + self.name() + self.argstr(True, split=True)
        else:
            return returntype + " " + self.name() + self.argstr(True)

    def print_call(self, namestringonly, indent=""):
        if namestringonly:
            return "(char*)\"" + self.name() + "\";\n"
        func_len = len(self.name() + self.argstr(False, indent) + ";\n")
        if func_len > self.max_len_of_line:
            return self.name() + self.argstr(False, indent, split=True) + ";\n"
        else:
            return self.name() + self.argstr(False, indent) + ";\n"

    def print_padding_condition(self, split=False):
        # we use 'SAME' padding scheme from the TensorFlow
        #the bottom and right sides may have the one additional padded pixel in some cases.
        #for example, an even size of a kernel and a stride equal to 1
        if (self.stride_w == 1) and (self.stride_h == 1):
            if self.padding == "krnpad":
                if (self.kernel_h > 0) and (self.kernel_w > 0):
                    cond  = "(padding_top == " + str(int((self.kernel_h - 1) / 2)) + ") && "
                    cond += "(padding_bot == " + str(int(self.kernel_h / 2)) + ") && "
                    cond += "(padding_left == " + str(int((self.kernel_w -1) / 2)) + ") && "
                    cond += "(padding_right == " + str(int(self.kernel_w / 2)) + ")"
                else:
                    cond = "(1)"
            elif self.padding == "nopad":
                cond  = "(padding_top == 0) && "
                cond += "(padding_bot == 0) && "
                cond += "(padding_left == 0) && "
                cond += "(padding_right == 0)"
            else:
                cond = "(1)"
        elif self.padding == "nopad":
            cond  = "(padding_top == 0) && "
            cond += "(padding_bot == 0) && "
            cond += "(padding_left == 0) && "
            cond += "(padding_right == 0)"
        elif self.padding == "krnpad" and (self.kernel_h > 0) and (self.kernel_w > 0):
            cond  = "(padding_top <= " + str(int((self.kernel_h - 1) / 2)) + ") && "
            cond += "(padding_bot <= " + str(int(self.kernel_h / 2)) + ") && "
            cond += "(padding_left <= " + str(int((self.kernel_w -1) / 2)) + ") && "
            cond += "(padding_right <= " + str(int(self.kernel_w / 2)) + ")"
        else:
            cond = "(1)"
        return cond

    def print_condition(self, split=False):
        indent = ""
        newline = ""
        cond_count = 0
        if (split):
            indent = " "*12
            newline = "\n"
        cond = ""
        sep = ""
        if (self.stride_w > 0):
            cond += sep + "(stride_w == " + str(self.stride_w) + ")"
            sep = " && "
            cond_count += 1
        if (self.stride_h > 0):
            cond += sep + "(stride_h == " + str(self.stride_h) + ")"
            sep = " && "
            cond_count += 1
        if (self.kernel_w > 0):
            cond += sep + newline + indent + "(kernel_w == " + str(self.kernel_w) + ")"
            sep = " && "
            cond_count += 1
        if (self.kernel_h > 0):
            cond += sep + "(kernel_h == " + str(self.kernel_h) + ")"
            sep = " && "
            cond_count += 1
        if (self.channels > 0):
            cond += sep + newline + indent + "(channels == " + str(self.channels) + ")"
            sep = " && "
            cond_count += 1
        if (self.print_padding_condition() != "(1)"):
            #skip padding
            cond += sep + newline + indent + self.print_padding_condition(split=split)
            cond_count += 1

        if (cond_count > 1):
            return "(" + cond + ")"
        else:
            return cond

    def get_types(self):
        if self.datatype == "fx16":
            d_type = "int16_t"
            w_type = "int16_t"
            b_type = w_type
            d_enum = "MLI_EL_FX_16"
            el_params = "fx.frac_bits"
            return (d_type, w_type, d_enum, el_params, b_type)
        if self.datatype == "fx8":
            d_type = "int8_t"
            w_type = "int8_t"
            b_type = w_type
            d_enum = "MLI_EL_FX_8"
            el_params = "fx.frac_bits"
            return (d_type, w_type, d_enum, el_params, b_type)
        if self.datatype == "fx8w16d":
            d_type = "int16_t"
            w_type = "int8_t"
            b_type = w_type
            d_enum = "MLI_EL_FX_16"
            el_params = "fx.frac_bits"
            return (d_type, w_type, d_enum, el_params, b_type)
        if self.datatype == "sa8":
            d_type = "int8_t"
            w_type = "int8_t"
            b_type = w_type
            d_enum = "MLI_EL_ASYM_I8"
            el_params = "asym"
            return (d_type, w_type, d_enum, el_params, b_type)
        if self.datatype == "sa8_sa8_sa32":
            d_type = "int8_t"
            w_type = "int8_t"
            b_type = "int32_t"
            d_enum = "MLI_EL_ASYM_I8"
            el_params = "asym"
            return (d_type, w_type, d_enum, el_params, b_type)
        print("ERROR: unsopported type: " + self.datatype)
    def print_body(self, template_file):
        f = open(template_file, "r")
        s = Template(f.read())
        d_type, w_type, d_enum, el_params, b_type = self.get_types()
        # we use 'SAME' padding scheme from the TensorFlow
        #the bottom and right sides may have the one additional padded pixel in some cases.
        #for example, an even size of a kernel and a stride equal to 1
        if (self.stride_w == 1) and (self.stride_h == 1):
            if self.padding == "krnpad":
                if (self.kernel_h > 0) and (self.kernel_w > 0):
                    #compute the padding values based on the kernel size, and fix the padding values
                    k_pad = 1
                    pad_top = int((self.kernel_h-1) / 2)
                    pad_bot = int(self.kernel_h / 2)
                    pad_left = int((self.kernel_w-1) / 2)
                    pad_right = int(self.kernel_w / 2)
                else:
                    # kernel_h or kernel_w not set.
                    #k_pad = 0 means that padding values are not fixed. so the values are don't care
                    k_pad = 0
                    pad_top = 0
                    pad_bot = 0
                    pad_left = 0
                    pad_right = 0
            elif self.padding == "nopad":
                #in case of 'nopad' we can fix the padding values to 0
                k_pad = 1
                pad_top = 0
                pad_bot = 0
                pad_left = 0
                pad_right = 0
            else:
                #k_pad = 0 means that padding values are not fixed. so the values are don't care
                k_pad = 0
                pad_top = 0
                pad_bot = 0
                pad_left = 0
                pad_right = 0
        elif self.padding == "nopad":
                #in case of 'nopad' we can fix the padding values to 0
                k_pad = 1
                pad_top = 0
                pad_bot = 0
                pad_left = 0
                pad_right = 0
        else:
            #k_pad = 0 means that padding values are not fixed. so the values are don't care
            k_pad = 0
            pad_top = 0
            pad_bot = 0
            pad_left = 0
            pad_right = 0

        namelist = ['core_name', 'core_name1', 'core_name2', 'core_name3',
                    'core_name4', 'core_name5', 'core_name6', 'core_name7',
                    'core_name8', 'core_name9']
        hasnamelist = ['has_core_name', 'has_core_name1', 'has_core_name2', 'has_core_name3',
                    'has_core_name4', 'has_core_name5', 'has_core_name6', 'has_core_name7',
                    'has_core_name8', 'has_core_name9']
        mapping = {}
        for hasname in hasnamelist:
            mapping[hasname] = 0

        for corefunc in self.corefunc:
            corename = namelist.pop(0)
            hasname = hasnamelist.pop(0)
            mapping[corename] = corefunc
            mapping[hasname] = 1

        return s.safe_substitute(mapping, name = self.name(),
                            kernel_w = self.kernel_w,
                            kernel_h = self.kernel_h,
                            channels = self.channels,
                            stride_w = self.stride_w,
                            stride_h = self.stride_h,
                            datatype = self.datatype,
                            el_params = el_params,
                            d_type = d_type,
                            w_type = w_type,
                            d_enum_type = d_enum,
                            b_type = b_type,
                            kernelpadding = k_pad,
                            padding_top = pad_top,
                            padding_bot = pad_bot,
                            padding_left = pad_left,
                            padding_right = pad_right)
