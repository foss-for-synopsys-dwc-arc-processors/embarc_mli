from string import Template
from func import Func

class Codegen:
    wrapper_variables = {}
    hierarchy = []
    tree = True
    previous_was_param = False

    def set_wrapper_if_tree(self, use_if_tree_or_if_list):
        self.tree = use_if_tree_or_if_list
    def get_param_range_from_list(self, func_list, param_name):
        param_list = []
        for f in func_list:
            param = f.get_param_by_name(param_name)
            if not param in param_list:
                param_list.append(param)
        return param_list

    def get_sublist(self, func_list, param_name, param):
        sub_list = []
        for f in func_list:
            par = f.get_param_by_name(param_name)
            if (par == param):
                sub_list.append(f)
        return sub_list

    def print_if_tree(self, func_list, hierargy_list, indent, default_func, namestringonly):
        string = ""
        if len(hierargy_list) == 0:
            return indent + "return " + func_list[0].print_call(namestringonly)
        local_hierargy_list = list(hierargy_list)
        param_name = local_hierargy_list.pop(0)
        param_range = self.get_param_range_from_list(func_list, param_name)
        sep = ""
        param_range.sort(reverse=True)
        has_else_branch = False
        for param in param_range:
            sub_list = self.get_sublist(func_list, param_name, param)
            if sep == "else ":
                if param:
                    if param_name == "padding":
                        if len(sub_list) > 1:
                            print "padding needs to be deepest hierarchy level otherwise the condition cannot be calculated"
                        if (sub_list[0].print_padding_condition() == "(1)"):
                            string = string[:len(string)-1] + " "
                            string += sep
                            has_else_branch = True
                        else:
                            string = string[:len(string)-1] + " "
                            string += sep + "if (" + sub_list[0].print_padding_condition() + ") "
                    else:
                        string = string[:len(string)-1] + " "
                        string += sep + "if ({0} == {1}) ".format(param_name, param)
                    string += "{\n"
                else:
                    string = string[:len(string)-1] + " "
                    string += sep + "{\n"
                    has_else_branch = True
                string += self.print_if_tree(sub_list, local_hierargy_list, indent + "    ", default_func, namestringonly)
                string += indent + "}\n"
            else:
                if param:
                    if param_name == "padding":
                        if len(sub_list) > 1:
                            print "padding needs to be deepest hierarchy level otherwise the condition cannot be calculated"
                        if (sub_list[0].print_padding_condition() == "(1)"):
                            string += indent + sep
                            has_else_branch = True
                        else:
                            string += indent + sep + "if (" + sub_list[0].print_padding_condition() + ") "
                    else:
                        string += indent + sep + "if ({0} == {1}) ".format(param_name, param)
                    string += "{\n"
                else:
                    string += indent + sep + "{\n"
                    has_else_branch = True
                string += self.print_if_tree(sub_list, local_hierargy_list, indent + "    ", default_func, namestringonly)
                string += indent + "}\n"
            sep = "else "
        if not has_else_branch:
            string = string[:len(string)-1] + " "
            string += sep + "{\n"
            string += indent + " "*4 + "return " + default_func.print_call(namestringonly)
            string += indent + "}\n"
        return string

    def sort_func_list(self, func_list, hierargy_list):
        if len(hierargy_list) == 0:
            return func_list
        local_hierargy_list = list(hierargy_list)
        param_name = local_hierargy_list.pop(0)
        param_range = self.get_param_range_from_list(func_list, param_name)
        param_range.sort(reverse=True)
        newlist = []
        for param in param_range:
            sub_list = self.get_sublist(func_list, param_name, param)
            newlist.extend(self.sort_func_list(sub_list, local_hierargy_list))
        return newlist

    def print_func_list(self, func_list):
        string = ""
        for func in func_list:
            string += func.name() + "()\n"
        return string

    def print_if_list(self, func_list, hierargy_list, default_func, namestringonly=False):
        string = ""
        sep = "    "
        sorted_list = self.sort_func_list(func_list, hierargy_list)
        for func in sorted_list:
            if func.generic:
                continue #call to generic should be done at the end
            string += sep + "if "
            cond = func.print_condition()
            if (len(cond) <= func.max_len_of_line):
                string += cond
            else:
                string += func.print_condition(split=True)
            string += " {\n"
            string += " "*8 + "return " + func.print_call(namestringonly)
            string += " "*4 + "}"
            sep = " else "
        string += " else {\n"
        string += " "*8 + "return " + default_func.print_call(namestringonly)
        string += " "*4 + "}\n"
        return string

    def print_func_bodies(self, func_list, template_file):
        string = ""
        for f in func_list:
            string += f.print_proto() + " {\n"
            string += f.print_body(template_file) + "\n"
        return string

    def print_func_proto(self, func_list):
        string = "\n"
        for f in func_list:
            new_str = f.print_proto() + ";\n"
            string += new_str
            if (len(new_str) > f.max_len_of_line):
                string += "\n"
        return string

    def set_wrapper_hierarchy(self, varlist):
        self.hierarchy = varlist

    def set_wrapper_variables(self, variables):
        self.wrapper_variables.update(variables)

    def print_variables(self):
        string = ""
        for var in self.hierarchy:
            if var == 'padding':
                try:
                    for padvar in ['padding_top', 'padding_bot', 'padding_left', 'padding_right']:
                        string += "    int " + padvar + " = " + self.wrapper_variables[padvar] + ";\n"
                except KeyError, e:
                    print 'Wrapper hierarchy contains a padding and set_wrapper_variables should have: "%s"' % str(e)
            else:
                try:
                    string += "    int " + var + " = " + self.wrapper_variables[var] + ";\n"
                except KeyError, e:
                    print 'Wrapper hierarchy contains a variable that is not provided in set_wrapper_variables: "%s"' % str(e)
        return string
    def print_wrapper(self, func_list, default_func, debugwrapper=False):
        base_func = func_list[0].get_base_func()
        #todo: check if all functions in the list have the same base
        if debugwrapper:
            returntype = "char *"
            base_func.debug = True
        else:
            returntype = "mli_status"
        string = base_func.print_proto(returntype=returntype) + " {\n"
        string += self.print_variables()
        string += "\n"
        if self.tree == True:
            string += self.print_if_tree(func_list, self.hierarchy, "    ", default_func, debugwrapper)
        else:
            string += self.print_if_list(func_list, self.hierarchy, default_func, debugwrapper)
        string += "}\n"
        return string
    def print_wrapper_proto(self, func_list, debugwrapper=False):
        base_func = func_list[0].get_base_func()
        if debugwrapper:
            returntype = "char *"
            base_func.debug = True
        else:
            returntype = "mli_status"
        string = base_func.print_proto(returntype=returntype) + ";\n"
        return string

    def print_includes(self, include_list):
        string = ""
        for inc in include_list:
            string += "#include \"" + inc + "\"\n"
        return string

    def print_define(self, define_list):
        string = ""
        for dfn in define_list:
            string += dfn + "\n"
        return string

    def print_file(self, func_list, default_func, body_template, file_template, include_list, define_list):
        f = open(file_template, "r")
        s = Template(f.read())
        string = "/* This file is generated, do not edit!\n"
        string += " * edit following template files instead:\n"
        string += " * " + file_template + "\n"
        string += " * " + body_template + "\n"
        string += " */\n"

        string += s.substitute(extra_includes = self.print_includes(include_list),
                               extra_defines = self.print_define(define_list),
                               functions = self.print_func_bodies(func_list, body_template),
                               wrapper = self.print_wrapper(func_list, default_func) + self.print_wrapper(func_list, default_func, debugwrapper=True))
        return string
    def print_proto_file(self, func_list_list, function_group, cap_header_file_name, header_file_template):
        f = open(header_file_template, "r")
        s = Template(f.read())
        string = "/* This file is generated, do not edit!\n"
        string += " * edit following template file instead:\n"
        string += " * " + header_file_template + "\n"
        string += " */\n"
        functions_list = ""
        wrapper_list = ""
        for f_list in func_list_list:
            functions_list += self.print_wrapper_proto(f_list, debugwrapper=True)
            functions_list += self.print_func_proto(f_list)
        string += s.substitute(capital_file_name = cap_header_file_name,
                               func_group = function_group,
                               functions = functions_list)
        return string
