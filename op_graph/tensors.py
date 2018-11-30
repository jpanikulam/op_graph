'''
Reading:
[1] http://web.mit.edu/edbert/GR/gr1.pdf
[2] http://www.matrixcalculus.org/matrixcalculus.pdf
'''

# Smoke:
from op_defs import op


def make_function(name, arguments, sexpr, commutative=False, associative=False):
    # Construct a function and a commuted version of the same function
    pass


def einsum(gr, a, b):
    pass


ricci_gemm = {
    ('matrix', 'matrix'): {
        # Add a function for curring special arguments
        'generate': lambda a, b: op('einsum', a, b, (('a', 'b'), ('b', 'c'), ('a', 'c'))),
        'gen': lambda a, b: op('einsum', a, b, ('ab,bc->ac'))

    }
}
