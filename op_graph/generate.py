from text import clang_fmt
import cc_types


def func_name(func):
    if func['member_of'] is None:
        return func['name']
    return "{}::{}".format(func['member_of'], func['name'])


class CodeBlock(object):
    def __init__(self, deps=[]):
        self.code = ""
        self._deps = list(deps)

    @property
    def deps(self):
        """Dependencies."""
        return self._deps

    def line(self, *args):
        ltext = ' '.join(map(str, args))
        self.code += '{ltext};'.format(ltext=ltext)

    def set(self, left, right):
        out = []
        if isinstance(left, (list, tuple)):
            out.extend(left)
        else:
            out.append(left)
        out.append('=')
        if isinstance(right, (list, tuple)):
            out.extend(right)
        else:
            out.append(right)

        self.line(*out)

    def write(self, text):
        self.code += str(text)

    def comment(self, text):
        for line in text.split('\n'):
            self.code += "// {}\n".format(line)

    def __str__(self):
        return clang_fmt(self.code)


class Scope(CodeBlock):
    def __init__(self, pretext=""):
        self.code = "{}".format(pretext)

    def __enter__(self):
        self.write('{')
        return self

    def __exit__(self, *args):
        self.write('}')


class StructScope(Scope):
    def __init__(self, name):
        super(StructScope, self).__init__(pretext="struct {}".format(name))

    def __exit__(self, *args):
        self.write('};')


def declare_struct(struct):
    with StructScope(struct['name']) as code:
        for member in struct['members']:
            lhs = "{} {}".format(
                cc_types.type_to_str(member['type']),
                member['name']
            )

            zero = cc_types.zero(member['type'])
            rhs = struct['default_values'].get(member['name'], zero)

            if rhs is not None:
                code.set(lhs, rhs)
            else:
                code.line(lhs)

        for member_function in struct['member_functions']:
            func_decl = "static " + declare_func(member_function)
            code.write(func_decl)

    return code.code


def generate_struct(struct):
    code = CodeBlock()
    for member_function in struct['member_functions']:
        # func_decl = "static {}::".format(struct['name']) + declare_func(member_function)
        # func_decl = "static " + declare_func(member_function)
        # code.write(func_decl)
        code.write(generate_func(member_function, namespace=struct['name']))
    return code.code


def generate_func(func, namespace=None):
    text = ""
    text += func_decl(func, namespace)
    with Scope() as code:
        if func['impl'] is not None:
            code.write(func['impl'])

    text += code.code
    return text


def generate(thing):
    generation_dispatch = {
        'struct': generate_struct,
        'function': generate_func,
    }
    dispatch = generation_dispatch[thing['kind']]
    return clang_fmt(dispatch(thing))


def func_decl(func, namespace=None):
    args_list = ['{} {}'.format(cc_types.type_to_str(arg['type']), arg['name']) for arg in func['args']]
    assert func['kind'] == 'function'

    args = ",".join(args_list)

    prefix = ""
    if namespace is not None:
        prefix = "{}::".format(namespace)

    return "{} {}({})".format(
        cc_types.type_to_str(func['returns']),
        prefix + func['name'],
        args
    )


def declare_func(func):
    return func_decl(func) + ';'


def declare(thing):
    declaration_dispatch = {
        'struct': declare_struct,
        'function': declare_func,
    }
    dispatch = declaration_dispatch[thing['kind']]
    return dispatch(thing)
