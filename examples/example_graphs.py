from op_graph import graph


def vectorspring():
    gr = graph.OpGraph('VectorSpring')
    k = gr.scalar('k')

    imass = gr.inv('imass', gr.scalar('mass'))

    a = gr.vector('a', 3)
    v = gr.time_antiderivative('v', a)
    x = gr.time_antiderivative('x', v)

    f = gr.mul('f', k, x)
    gr.mul('a', imass, f)
    return gr


def controlled_vectorspring():
    gr = graph.OpGraph('VectorSpring')
    k = gr.scalar('k')

    imass = gr.inv(gr.anon(), gr.scalar('mass'))

    u = gr.vector('u', 3)

    a = gr.vector('a', 3)
    v = gr.time_antiderivative('v', a)
    x = gr.time_antiderivative('x', v)

    force = gr.add('force', gr.mul(gr.anon(), k, x), u)
    gr.mul('a', imass, force)
    return gr


def simple_graph():
    gr = graph.OpGraph('Simple')

    a = gr.scalar('a')
    gr.optimize(a)
    b = gr.scalar('b')
    gr.mul('ab', a, b)
    d = gr.time_antiderivative('d', 'ab')
    gr.time_antiderivative('e', d)
    return gr


def double_integrator():
    gr = graph.OpGraph('double_integrator')
    gr.scalar('u')
    gr.optimize('u')
    gr.time_antiderivative('v', 'u')
    gr.time_antiderivative('x', 'v')
    return gr


def rotary_double_integrator():
    gr = graph.OpGraph('double_integrator')
    gr.vector('u', 3)
    gr.optimize('u')
    gr.time_antiderivative('w', 'u')
    gr.so3('R')
    gr.time_antiderivative('R', 'w')
    return gr


all_graphs = [
    vectorspring(),
    simple_graph(),
    double_integrator(),
    rotary_double_integrator(),
    controlled_vectorspring()
]


def main():
    for gr in all_graphs:
        print gr


if __name__ == '__main__':
    main()
