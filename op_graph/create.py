def create_lvalue(cpp_type, name):
    return {
        'kind': 'lvalue',
        'type': cpp_type,
        'name': name
    }


def create_struct(name, members):
    return {
        'kind': 'struct',
        'name': name,
        'members': members,
    }


def create_function(name, arguments, returns, impl=None):
    return {
        'kind': 'function',
        'name': name,
        'args': arguments,
        'returns': returns,
        'impl': str(impl) if impl is not None else None
    }
