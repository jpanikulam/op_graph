from op_graph import graph
from op_graph import graph_to_cc
from op_graph import integration
from op_graph.code import CodeGraph

# Register a function that someone will implement in C++
# gr.add_function(
#     'force_from_throttle',
#     returns=graph.create_scalar(),
#     arguments=(
#         graph.create_scalar(),   # Throttle
#         graph.create_scalar(),   # Temperature
#         graph.create_vector(3),  # Airspeed (Just to throw a vector in)
#     )
# )


def make_force_fcn():
    gr = graph.OpGraph('ForceFromThrottle')
    gr.scalar('throttle')
    gr.identity('out', 'throttle')
    return gr


def make_simple_jet():
    gr = graph.OpGraph()

    gr.scalar('mass')
    gr.vector('external_force', 3)

    gr.so3('R_world_from_body')
    gr.optimize(gr.vector('q', 3))

    gr.time_antiderivative('w', 'q')

    # TODO: Add subtraction so we can do damping
    # gr.mul('damped_w', 'w_damping', 'w')
    # gr.add('')

    gr.time_antiderivative('R_world_from_body', 'w')

    # Or make a dummy we can use
    gr.add_graph_as_function('force_from_throttle', make_force_fcn(), 'out')

    gr.optimize(gr.scalar('throttle_dot'))

    gr.time_antiderivative('throttle_pct', 'throttle_dot')
    gr.func('force_from_throttle', 'thrust', 'throttle_pct')

    gr.vector('unit_z', 3)
    gr.mul('force_world', 'R_world_from_body', gr.mul('body_force', 'thrust', 'unit_z'))

    gr.add('net_force_world', 'force_world', 'external_force')

    gr.mul('a', gr.inv('inv_mass', 'mass'), 'net_force_world')

    gr.vector('v', 3)
    gr.time_antiderivative('v', 'a')
    gr.time_antiderivative('x', 'v')

    gr.warnings()
    return gr


def main():
    jet_graph = make_simple_jet()
    print jet_graph
    rk4 = integration.rk4_integrate(jet_graph)

    cg = CodeGraph(name='integrator', namespaces=['planning', 'jet'])
    graph_to_cc.express(cg, rk4)

    root = '/home/jacob/repos/experiments/'
    loc = 'planning/jet'
    name = 'jet_dynamics'

    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    main()
