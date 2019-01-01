import cc_types


def create_type(cpp_type):
    return cc_types.typen(cpp_type)


def create_lvalue(cpp_type, name):
    created_type = create_type(cpp_type)
    return {
        'kind': 'lvalue',
        'type': created_type,
        'name': name,
    }


def create_struct(name, members, default_values={}, member_functions=[]):
    dependencies = []
    for mem in members:
        dependencies.append(mem['type'])

    return {
        'kind': 'struct',
        'name': name,
        'members': members,
        'member_functions': member_functions,
        'default_values': default_values,
        'deps': dependencies
    }


def create_function(name, arguments, returns, member_of=None, impl=None):
    if impl is None:
        impl_deps = []
    else:
        impl_deps = impl.deps

    dependencies = []
    for mem in arguments:
        dependencies.append(mem['type'])
    deps = dependencies + impl_deps

    return {
        'kind': 'function',
        'name': name,
        'args': tuple(arguments),
        'returns': returns,
        'impl': str(impl) if impl is not None else None,
        'deps': deps,
        'member_of': None,
    }
