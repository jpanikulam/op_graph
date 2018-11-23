from graph import OpGraph


def grouptest():
    a = ['a0', 'a1', 'a2']
    b = ['b0', 'b1', 'b2']
    m = ['m0', 'm1', 'm2']
    s = ['s0', 's1', 's2']

    gr = OpGraph()

    for aa in a:
        gr.vector(aa, 3)

    for bb in b:
        gr.vector(bb, 3)

    for mm in m:
        gr.scalar(mm)

    for ss in s:
        gr.so3(ss)

    A = gr.groupify("A", a)
    B = gr.groupify("B", b)
    M = gr.groupify("M", m)
    S = gr.groupify("S", s)

    print gr.add("C", A, B)
    gr.add("D", "C", B)
    gr.mul("E", M, "D")
    gr.mul("R", S, "D")

    gr.extract('r1', "R", 1)

    print gr


def functest():
    gr = OpGraph()

    gr.scalar('super_density')
    gr.scalar('mass')
    gr.so3('R')

    gr.vector('q', 3)

    gr.mul('density', 'mass', 'super_density')
    gr.mul('a', gr.inv('inv_density', 'density'), 'q')
    gr.mul('Ra', 'R', 'a')

    # Broken -- Define op
    gr.add('RRa', 'R', 'Ra')
    gr.time_antiderivative('R', 'q')

    gr.groupify('Out', ['a', 'Ra', 'R'])

    gr2 = OpGraph()
    gr2.insert_subgraph(gr, 'Out', up_to=['inv_density'])

    gr3 = OpGraph()
    gr3.add_graph_as_function('poopy_func', gr2, 'Out')

    gr3.scalar('mass')
    gr3.vector('u', 3)

    gr3.func('poopy_func', 'rxx', 'u', 'mass')
    gr3.constant_scalar('half', 0.5)

    print gr
    print gr2
    print gr3._subgraph_functions
    print gr3

    print gr3.adj['rxx']
    print gr3.get_properties('rxx')


if __name__ == '__main__':
    functest()
    grouptest()
