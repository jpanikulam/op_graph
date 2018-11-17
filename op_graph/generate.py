from text import clang_fmt


class CodeBlock(object):
    def __init__(self):
        self.code = ""

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


def generate_struct(struct):
    with StructScope(struct['name']) as code:
        for member in struct['members']:
            code.line(member['type']['name'], member['name'])
    return code.code


def generate_func(func):
    text = ""
    text += declare(func)
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


def declare_struct(struct):
    return generate(struct)


def declare_func(func):
    args_list = ['{} {}'.format(arg['type']['name'], arg['name']) for arg in func['args']]
    assert func['kind'] == 'function'

    args = ",".join(args_list)
    return "{} {}({})".format(
        func['returns']['name'],
        func['name'],
        args
    )


def declare(thing):
    declaration_dispatch = {
        'struct': declare_struct,
        'function': declare_func,
    }
    dispatch = declaration_dispatch[thing['kind']]
    return dispatch(thing)
