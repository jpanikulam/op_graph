import generate
import create
import graph
import graph_tools

from functools import partial
from copy import deepcopy


def to_cpp_type(properties):
    choices = {
        'liegroup': properties['subtype'] if 'subtype' in properties.keys() else '',
        'vector': 'VecNd<{}>'.format(properties['dim']),
        'scalar': 'double',
    }
    return choices[properties['type']]


def to_const_ref(my_type):
    new_type = deepcopy(my_type)
    new_type['type'] = "const {}&".format(my_type['type'])
    return new_type


def integrate(sym, gr, gr_new):
    op = gr.adj[sym]
    sym_prop = gr.properties[sym]

    assert op[0] == 'time_antiderivative'
    derivative = graph.get_args(op)[0]

    def old(name):
        return 'old_{}'.format(name)

    def new(name):
        return 'new_{}'.format(name)

    derivative_old = gr_new.emplace(old(derivative), gr.properties[derivative])
    gr_new.emplace(old(sym), gr.properties[sym])

    product = gr_new.mul('{}_dt'.format(derivative_old), 'dt', derivative_old)
    # Integrate normally if not liegroup
    if sym_prop['type'] != 'liegroup':
        gr_new.add(new(sym), old(sym), product)
    else:
        # Integrate via exponentiation if liegroup
        group = sym_prop['subtype']
        exp_delta = gr_new.exp('exp_{}'.format(product), product, kind=group)
        gr_new.mul(new(sym), exp_delta, old(sym))


def to_trivial_integrator(gr):
    gr_new = graph.OpGraph()
    states = graph.get_states(gr)

    gr_new.scalar('dt')

    for sym in states:
        op = gr.adj[sym]
        sym_prop = gr.properties[sym]
        integrate(sym, gr, gr_new)

    print gr_new


def get_dependency_subgraph(gr, sym, up_to=[], excludes=[]):
    deps = []
    op = gr.adj[sym]

    if op is None or op[0] in excludes:
        return deps

    for arg in graph.get_args(op):
        deps.append(arg)
        if arg in up_to:
            continue

        arg_deps = get_dependency_subgraph(gr, arg, up_to=up_to, excludes=excludes)
        deps.extend(arg_deps)
    return deps


def rk4_integrate(gr):
    """How do we compute this.

    qdot is an output subgraph of gr
    qdot is a function of some subset of the inputs of gr, including q

    k1 = qdot(q)
    k2 = qdot(q + (h / 2) * k1)
    k3 = qdot(q + (h / 2) * k2)
    k4 = qdot(q + h * k2)

    h2 = h * 2
    dq = (1 / 6) * h2 * (k1 + 2 * (k2 + k3) + k4)
    q1 = q + dq

    qdot: states, controls; side_inputs -> dq_dt
    qdot: x, u; z -> dq_dt

    """
    # Compute qdot
    qdot_gr = graph.OpGraph()
    qdot = set()

    states = graph.get_states(gr)

    for state in states:
        # All states are inputs.
        qdot_gr.emplace(state, gr.properties[state])

    for state in states:
        statedot = graph.get_args(gr.adj[state])[0]
        qdot.add(statedot)

        qdot_gr.insert_subgraph(gr, statedot, up_to=states)
        qdot_gr.identity('{}_dot'.format(state), statedot)

    print '\nx:'
    print states
    print '\nu:'
    print gr.to_optimize
    print '\nz:'
    print graph.get_inputs(gr).intersection(graph.get_inputs(qdot_gr)) - gr.to_optimize
    print '\nqdot:'
    print qdot

    qdot_group = gr.groupify('Qdot', qdot)
    states = gr.groupify('Q', tuple(states))

    print states
    print qdot_group

    print gr

    # k1 = deepcopy(qdot_gr)
    # k2 = deepcopy(qdot_gr)
    h = k2.scalar('h2')

    repls = {}
    for state in qdot:
        # statedot = graph.get_args(k2.adj[state])[0]
        h2p = k2.mul('p1_{}'.format(state), h, state)
        fin = k2.add('k2_{}'.format(state), state, h2p)
        # repls[state] = fin
        break

    # k2.subs(repls)

    print '-----'
    print 'k1----'
    print k1
    print 'k2----'
    print k2

    # Re-evaluate qdot_gr(x=(h / 2) * x)


def binary(op, gr, operator=None):
    assert operator is not None
    args = graph.get_args(op)
    return "{a} {operator} {b}".format(
        a=args[0],
        b=args[1],
        operator=operator,
    )


def func(op, gr):
    args = graph.get_args(op)
    return "{}({})".format(op[0], ', '.join(args))


def inv(op, gr):
    args = graph.get_args(op)
    arg_type = gr.properties[args[0]]
    if arg_type == 'liegroup':
        return "{}.inverse()".format(args[0])
    else:
        return "(1.0 / {})".format(args[0])


def op_to_cc(op, gr):
    dispatch = {
        'add': partial(binary, operator='+'),
        'mul': partial(binary, operator='*'),
        'exp': func,
        'log': func,
        'inv': inv,
        'I': lambda op, gr: graph.get_args(op)[0],
    }

    result = dispatch[op[0]](op, gr)
    return result


def express(gr):
    gr_sorted = graph_tools.topological_sort(gr.adj)

    cb = generate.CodeBlock()

    for sym in gr_sorted:
        op = gr.adj[sym]
        if op is not None:
            sym_prop = gr.properties[sym]
            decl = "const {type} {name}".format(
                type=to_cpp_type(sym_prop),
                name=sym
            )
            cb.set(decl, op_to_cc(op, gr))

    print cb


def test_graph():
    gr = graph.OpGraph()

    a = gr.scalar('a')
    b = gr.scalar('b')
    gr.mul('ab', a, b)
    d = gr.time_antiderivative('d', 'ab')
    gr.time_antiderivative('e', d)
    print gr

    # Unoptimized
    # gr.so3('e')
    # a = gr.vector('a', 3)
    # b = gr.vector('b', 3)
    # d = gr.time_antiderivative('d', gr.mul('ec', 'e', gr.add('c', a, b)))
    # gr.time_antiderivative('e', d)
    # print gr

    # df_dt = gr.vector('df', 3)
    # f = gr.time_antiderivative('f', df_dt)

    # m = gr.scalar('m')

    # inv_m = gr.inv('inv_mass', m)
    # a = gr.mul('a', inv_m, f)
    # v = gr.time_antiderivative('v', a)
    # gr.time_antiderivative('x', v)

    '''
    gr.scalar('mass')
    gr.vector('popoy', 3)

    gr.so3('R_world_from_body')
    gr.optimize(gr.vector('q', 3))

    gr.mul('Rq', 'R_world_from_body', 'q')
    gr.mul('mRq', gr.inv('inv_mass', 'mass'), 'Rq')
    gr.mul('dRq', 'density', 'Rq')
    gr.add('popoy_dRq', 'popoy', 'dRq')

    gr.time_antiderivative('w', 'q')
    gr.time_antiderivative('R_world_from_body', 'w')
    '''

    # gr.mul('Rwbw', 'R_world_from_body', 'w')
    # gr.mul('Rwbwq', 'R_world_from_body', gr.exp('exp_w', 'w', kind='SO3'))

    return gr


def main():
    gr = test_graph()
    rk4_integrate(gr)
    # express(gr)


if __name__ == '__main__':
    main()
