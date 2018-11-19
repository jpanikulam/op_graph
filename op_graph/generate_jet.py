from graph import OpGraph, create_scalar, create_vector
import graph_to_cc
import integration


def make_simple_jet():
    graph = OpGraph()

    graph.scalar('density')
    # Unoptimized
    graph.scalar('mass')

    graph.so3('R_world_from_body')
    graph.optimize(graph.vector('q', 3))
    graph.time_antiderivative('w', 'q')
    graph.time_antiderivative('R_world_from_body', 'w')

    graph.add_function(
        'force_from_throttle',
        returns=create_scalar(),
        arguments=(
            create_scalar(),
            create_vector(3),
        )
    )

    graph.optimize(graph.scalar('throttle_dot'))

    graph.vector('v', 3)
    graph.time_antiderivative('throttle_pct', 'throttle_dot')
    graph.func('force_from_throttle', 'thrust', 'throttle_pct', 'v')

    graph.vector('unit_z', 3)
    graph.mul('force_world', 'R_world_from_body', graph.mul('body_force', 'thrust', 'unit_z'))

    graph.mul('a', graph.inv('inv_mass', 'mass'), 'force_world')

    graph.time_antiderivative('v', 'a')
    graph.time_antiderivative('x', 'v')

    graph.warnings()
    return graph


def main():
    jet_graph = make_simple_jet()

    rk4 = integration.rk4_integrate(jet_graph)

    graph_to_cc.express(rk4)
    # gd.make_types(jet_graph)
    # gd.generate_dynamics(jet_graph)


if __name__ == '__main__':
    main()
