import cc_types


def typen(text):
    template_args


def create_type(cpp_type):
    return {
        'kind': 'type',
        'name': cpp_type,
        'deps': [cc_types.type_dep(cpp_type)],
    }


def create_lvalue(cpp_type, name):
    created_type = create_type(cpp_type)
    return {
        'kind': 'lvalue',
        'type': created_type,
        'name': name,
        'deps': created_type['deps'],
    }


def create_struct(name, members):
    return {
        'kind': 'struct',
        'name': name,
        'members': members,
        'deps': list(map(lambda o: o['deps'], members)),
    }


def create_function(name, arguments, returns, impl=None):
    return {
        'kind': 'function',
        'name': name,
        'args': tuple(arguments),
        'returns': returns,
        'impl': str(impl) if impl is not None else None,
        'deps': list(map(lambda o: o['deps'], arguments)) + impl.deps
    }
