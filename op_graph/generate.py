from text import clang_fmt


class CodeBlock(object):
    def __init__(self, deps=[]):
        assert isinstance(deps, list)
        self.code = ""
        self._deps = deps

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


def type_to_str(type_data):
    template = ""
    if len(type_data['template_args']) > 0:
        template = "<{}>".format(", ".join(type_data['template_args']))

    qualifiers = " ".join(type_data['qualifiers'])
    postpenders = {
        'ref': '&',
        'ptr': '*'
    }

    postpend = ""
    for notion, token in postpenders.items():
        if type_data[notion]:
            postpend += token

    end_str = "{qualifiers} {type_name}{template}{postpender}".format(
        qualifiers=qualifiers,
        type_name=type_data['name'],
        template=template,
        postpender=postpend
    )

    return end_str


def generate_struct(struct):
    with StructScope(struct['name']) as code:
        for member in struct['members']:
            code.line(type_to_str(member['type']), member['name'])
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
    args_list = ['{} {}'.format(type_to_str(arg['type']), arg['name']) for arg in func['args']]
    assert func['kind'] == 'function'

    args = ",".join(args_list)
    return "{} {}({})".format(
        type_to_str(func['returns']),
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
