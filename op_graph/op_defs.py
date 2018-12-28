def first(f):
    return lambda a, b: f(a)


def second(f):
    return lambda a, b: f(b)


def shortcut(value):
    return lambda a, b: value


def create_constant(value, properties):
    """TODO is this the right approach?"""
    properties['constant'] = True
    properties['value'] = value
    return properties


def create_scalar():
    prop = {
        'type': 'scalar',
        'dim': (1, 1)
    }
    return prop


def create_vector(dim):
    if isinstance(dim, int):
        return create_matrix((dim, 1))
    elif isinstance(dim, tuple) and len(dim) == 2:
        assert dim[1] == 1
        return create_matrix(dim)
    else:
        raise TypeError("Invalid vector dimensions: {}".format(dim))


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
        'SO3': {'dim': (3, 1), 'algebra_dim': (3, 1)},
        'SE3': {'dim': (3, 1), 'algebra_dim': (6, 1)},
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


def op_table(gr):
    """TODO: Use this instead of anything else."""
    return {
        'inv': inv(gr),
        'mul': mul(gr),
        'add': add(gr),
        'sub': sub(gr),
        'exp': exp(gr),
        'log': log(gr),
    }


def d_table(gr):
    return {
        'mul': dmul(gr),
        'add': dadd(gr),
    }


class Constant(object):
    def __init__(self, properties, value):
        self._properties = properties

        value_remap = {
            0: 'zero',
            0.0: 'zero',
            1: 'I',
        }
        new_value = value_remap.get(value, value)

        valid_special_values = {
            'matrix': ('I', 'zero', 'zeros', 'ones'),
            'liegroup': ('I'),
            'scalar': ('I', 'zero'),
        }

        required_type = {
            'matrix': (tuple, list),
            'liegroup': (tuple, list),
            'scalar': (float, int),
        }

        value_type = properties['type']
        if value_type == 'matrix':
            dim = properties['dim']
            if new_value in ['I', 'zero']:
                assert dim[0] == dim[1], "Zero and identity must be square (Did you mean 'zeros'?)"
            elif new_value in ['zeros', 'ones']:
                assert dim[1] == 1, "Zeros and Ones must be vectors"

        if isinstance(new_value, str):
            assert new_value in valid_special_values[value_type]
        else:
            assert isinstance(new_value, required_type[value_type])
        self._value = new_value

    @property
    def properties(self):
        return self._properties

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        return self._properties == other._properties and self._value == other._value

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return str(self._value)


def op(name, *args):
    return (name, tuple(args))


def inv(gr):
    return {
        ('liegroup',): {
            'returns': gr._inherit_last,
            'needs': [],
            'inverse': 'inv'
        },
        ('matrix',): {
            'returns': gr._inherit_last,
            'needs': [],
            'inverse': 'inv'
        },
        ('scalar',): {
            'returns': gr._inherit_last,
            'needs': [],
            'inverse': 'inv'
        },
    }


def da_db(gr, a, b):
    p_a = gr.get_properties(a)
    p_b = gr.get_properties(b)

    xdim = gr.cross_dim(b, a)
    if p_a['dim'] and p_b['dim']:
        pass

    return


def add(gr):
    return {
        ('liegroup', 'matrix'): {
            'returns': gr._inherit_first,
            'needs': [gr._needs_valid_derivative_type],
            'generate': lambda n, a, b: gr._call('mul', n, gr._anony_call('exp', b), a),
            'properties': [],
        },
        ('matrix', 'matrix'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['commutative', 'associative'],
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 0),
                lambda a, b: Constant(gr.get_properties(b), 0),
            ),
            'inverse': 'sub'
        },
        ('scalar', 'scalar'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['commutative', 'associative'],
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 0),
                lambda a, b: Constant(gr.get_properties(b), 0),
            ),
            'inverse': 'sub'
        },
    }


def dadd(gr):
    return {
        ('liegroup', 'vector'): (
            {'generate': lambda a, b: a / a}  # Must Fail
        ),
        ('vector', 'vector'): (
            {'generate': lambda a, b: (gr.identity_for(a))},
            {'generate': lambda a, b: (gr.identity_for(b))},
        ),
        ('scalar', 'scalar'): (
            {'generate': lambda a, b: Constant(create_scalar(), 1)},
            {'generate': lambda a, b: Constant(create_scalar(), 1)},
        ),

    }


def sub(gr):
    return {
        ('liegroup', 'liegroup'): {
            'returns': gr._inherit_first,
            'needs': [gr._needs_same],
            'generate': lambda n, a, b: gr._call('mul', n, a, gr._anony_call('inv', b)),
            'properties': [],
        },
        ('matrix', 'matrix'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['commutative', 'associative'],
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 0),
                lambda a, b: Constant(gr.get_properties(b), 0),
            ),
            'inverse': 'sub'
        },
        ('scalar', 'scalar'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['commutative', 'associative'],
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 0),
                lambda a, b: Constant(gr.get_properties(b), 0),
            ),
            'inverse': 'sub'
        },
    }


def mul(gr):
    # How do we generate *template* types?
    # Eesh, does this mean we have to define a template system?
    return {
        ('liegroup', 'matrix'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same_dim],
            'properties': [],
            'inverse': 'inv',
        },
        ('liegroup', 'liegroup'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['associative'],
            'inverse': 'inv',
            # How do we define the identity for this thing?
            # 'identity': Constant(gr, create_liegroup())
        },
        ('matrix', 'matrix'): {
            'returns': gr._matrix_mul_type,
            'needs': [gr._needs_valid_matmul],
            'properties': ['associative'],
            'inverse': 'inv',
            'identity': (
                lambda a, b: gr.identity_for(b),
                lambda a, b: gr.identity_for(a, right=True),
            )
        },
        ('scalar', 'matrix'): {
            'returns': second(gr._inherit),
            'needs': [],
            'properties': [],
            'inverse': 'inv',
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 1),
                # lambda a, b: gr.identity_for(b, right=True),
                None
            ),
            'zero': (
                # lambda a, b: Constant(gr.get_properties(a), 1),
                None,
                lambda a, b: gr.zeros_for(b, right=True),
            )
        },
        ('matrix', 'scalar'): {
            'returns': first(gr._inherit),
            'needs': [],
            'properties': [],
            'inverse': 'inv',
            'identity': (
                lambda a, b: gr.identity_for(b),
                lambda a, b: Constant(gr.get_properties(a), 1),
            ),
            'zero': (
                lambda a, b: Constant(gr.get_properties(a), 1),
                lambda a, b: gr.zeros_for(b),
            )
        },
        ('scalar', 'scalar'): {
            'returns': gr._inherit_last,
            'needs': [gr._needs_same],
            'properties': ['commutative', 'associative'],
            'inverse': 'inv',
            'identity': (
                lambda a, b: Constant(gr.get_properties(a), 1),
                lambda a, b: Constant(gr.get_properties(b), 1),
            ),
            'zero': (
                lambda a, b: Constant(gr.get_properties(a), 0),
                lambda a, b: Constant(gr.get_properties(b), 0),
            ),
            'identities': (
            )
        },
    }


def dmul(gr):
    """Generate derivatives for multiplication."""
    return {
        ('liegroup', 'matrix'): (
            {
                'generate': lambda a, b: op('negate', op('skew', op('mul', a, b)))
            },
            {
                'generate': lambda a, b: b,
                'needs': [gr._needs_vector],
            }
        ),
        ('liegroup', 'liegroup'): (
            {
                'generate': lambda a, b: a
            },
            {
                'generate': lambda a, b: op('Adj', a)
            }
        ),
        ('matrix', 'matrix'): (
            {
                # 'generate': lambda a, b: a
                'generate': first(gr.zeros_for),
                'needs': [first(gr._needs_constant)],
            },
            {
                'generate': lambda a, b: a,
                'needs': [second(gr._needs_vector)],
            }
        ),
        ('scalar', 'matrix'): (
            {
                'generate': lambda a, b: b
            },
            {
                'generate': lambda a, b: op('mul', a, gr.identity_for(b)),
                'needs': [second(gr._needs_vector)],
            }
        ),
        ('matrix', 'scalar'): (
            {
                'generate': lambda a, b: op('mul', b, gr.identity_for(a)),
                'needs': [second(gr._needs_vector)],
            },
            {
                'generate': lambda a, b: b
            }
        ),
        ('scalar', 'scalar'): (
            {
                'generate': lambda a, b: b
            },
            {
                'generate': lambda a, b: a
            }

        ),
    }


def exp(gr):
    return {
        ('matrix',): {
            'returns': gr.exp_type,
            'needs': []
        },
        # ('scalar',): {
            # 'returns': lambda a: create_scalar(),
            # 'needs': []
        # }
    }


def log(gr):
    return {
        ('liegroup',): {
            'returns': gr.derivative_type,
            'needs': []
        },
    }


def main():
    a = Constant(create_scalar(), 'I')
    b = Constant(create_scalar(), 'I')
    assert a == b

if __name__ == '__main__':
    main()
