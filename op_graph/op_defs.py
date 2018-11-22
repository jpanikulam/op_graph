def create_constant(value, properties):
    """TODO is this the right approach?"""
    properties['constant'] = True
    properties['value'] = value
    return properties


def create_scalar():
    prop = {
        'type': 'scalar',
        'dim': 1
    }
    return prop


def create_vector(dim):
    assert isinstance(dim, int)
    prop = {
        'type': 'vector',
        'dim': dim
    }
    return prop


def create_matrix(dim):
    assert isinstance(dim, tuple)
    assert len(dim) == 2
    prop = {
        'type': 'matrix',
        'dim': dim
    }
    return prop


VALID_LIEGROUPS = []


def create_liegroup(subtype):
    VALID_LIEGROUPS.append(subtype)
    mapping = {
        'SO3': {'dim': 3, 'algebra_dim': 3},
        'SE3': {'dim': 3, 'algebra_dim': 6},
    }

    default = {
        'type': 'liegroup',
        'subtype': subtype
    }

    default.update(mapping[subtype])
    return default


def create_group(element_properties, names, inherent_type=None):
    assert isinstance(element_properties, (list, tuple))
    assert isinstance(names, (list, tuple))
    assert len(names) == len(element_properties)
    return {
        'type': 'group',
        'elements': tuple(element_properties),
        'inherent_type': inherent_type,
        'names': tuple(names)
    }


def create_SO3():
    return create_liegroup('SO3')


def create_SE3():
    return create_liegroup('SE3')


def inv(gr):
    return {
        ('liegroup',): {
            'returns': gr._inherit_last,
            'needs': [],
        },
        ('scalar',): {
            'returns': gr._inherit_last,
            'needs': [],
        }
    }


def add(gr):
    return {
        ('liegroup', 'vector'): {
            'returns': gr._inherit_first,
            'needs': [gr._needs_valid_derivative_type],
            'generate': lambda n, a, b: gr._call('mul', n, gr._anony_call('exp', b), a)
        },
        ('vector', 'vector'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same]
        },
        ('scalar', 'scalar'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same]
        },
    }


def mul(gr):
    # How do we generate *template* types?
    # Eesh, does this mean we have to define a template system?
    return {
        ('liegroup', 'vector'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same_dim]
        },
        ('liegroup', 'liegroup'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same]
        },
        ('scalar', 'vector'): {
            'returns': gr._inherit_last,
            'needs': []
        },
        ('scalar', 'scalar'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same]
        },
    }


def dmul(gr):
    """Generate derivatives for multiplication."""
    return {
        ('liegroup', 'vector'): (
            {
                'generate': lambda a, b: ('negate', ('skew', ('mul', a, b)))
            },
            {
                'generate': lambda a, b: ('identity', b)
            }
        ),
        ('liegroup', 'liegroup'): (
            {
                'generate': lambda a, b: ('identity', a)
            },
            {
                'generate': lambda a, b: ('Adj', a)
            }
        ),
        ('scalar', 'vector'): (
            {
                'generate': lambda a, b: ('identity', b)
            },
            # {
            #     'generate': lambda a, b: ('matrix_identity', b)
            # }
        ),
        ('scalar', 'scalar'): (
            {
                'generate': lambda a, b: ('identity', b)
            },
            {
                'generate': lambda a, b: ('identity', a)
            }

        ),
    }


def exp(gr):
    return {
        ('vector',): {
            'returns': gr.exp_type,
            'needs': []
        },
        ('scalar',): {
            'returns': lambda a: create_scalar(),
            'needs': []
        }
    }
