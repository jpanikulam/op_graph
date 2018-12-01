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


"""
Notes:
    - Clear distinction between vectors and covectors may be unnecessary?
    - Could treat everything as a tensor of rank (n, n)
        - And then two-forms/bilinear forms are just matrices

vector: vec(a)
matrix: (cov(a), vec(b))

bilinear form: (vec(a), vec(b))
two-form: (cov(a), cov(b))

>>> a = Tensor([3])
>>> b = Tensor([3])
>>> c = rmul(a, b)
    -> Tensor([3])
    == (a^k * b^k) -> c^k


>>> c = Tensor([], [3])
>>> d = rmul(a, c)
    -> Scalar()
    == (a^k * c_k) -> d

>>> D = Tensor([3], [3])
>>> E = Tensor([3], [3])
>>> F = rmul(D, E)
    -> Tensor([3], [3])
    == (D^a_b * E^b_c) -> F^a_c

(Am I applying rank correctly here?)
>>> A = Tensor([], [3, 3])
>>> x = Tensor([3])
>>> g = rmul(A, x)
    -> g = Tensor([], [3])
    == (A_ab * x^b) -> g_a
>>> c = rmul(x, g)
    -> c = Scalar()
    == (g_a * x^a) -> c

dc/dx^a = (dc/dg_b * dg^b/dx_a) + dc/dx^a
    dc/dg = x^a
    dg_a/dx^b = A_ab
    dc/dx^a = g_a

    dc/dx^a = (A_ab * x^a) + g_a
     = (A_ab * x^a) + A_ab * x^a
     = 2 * A_ab * x^a

"""


ricci_gemm = {
    ('matrix', 'matrix'): {
        # Add a function for curring special arguments
        'generate': lambda a, b: op('einsum', a, b, (('a', 'b'), ('b', 'c'), ('a', 'c'))),
        'gen': lambda a, b: op('einsum', a, b, ('ab,bc->ac'))

    }
}
