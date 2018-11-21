import cc_types


def create_type(cpp_type):
    return cc_types.typen(cpp_type)


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
    if impl is None:
        impl_deps = []
    else:
        impl_deps = impl.deps
    deps = list(map(lambda o: o['deps'], arguments)) + impl_deps

    return {
        'kind': 'function',
        'name': name,
        'args': tuple(arguments),
        'returns': returns,
        'impl': str(impl) if impl is not None else None,
        'deps': deps
    }
