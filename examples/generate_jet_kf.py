from op_graph import graph
from op_graph import graph_to_cc
from op_graph import integration
from op_graph.code import CodeGraph
from op_graph import groups


def imu_observation_model():
    gr = graph.OpGraph()
    world_from_sensor = gr.se3('T_world_from_sensor')

    eps_dot = gr.vector('eps_dot', 6)
    eps_ddot = gr.vector('eps_ddot', 6)

    w = gr.block('w', eps_dot, 0, 0, 3, 1)
    v = gr.block('v', eps_dot, 3, 0, 3, 1)

    a = gr.block('a', eps_ddot, 0, 0, 3, 1)
    q = gr.block('q', eps_ddot, 3, 0, 3, 1)

    adj = gr.adjoint('adj', world_from_sensor)
    accel_terms = gr.mul('accel_terms', adj, 'eps_ddot')

    R = gr.block('R', adj, 0, 0, 3, 3)
    Rvx = gr.cross_matrix('Rvx', gr.mul(gr.anon(), R, v))
    Rvxw = gr.mul(gr.anon(), Rvx, w)

    centrifugal = gr.cross(gr.anon(), )

    velocity_terms =

    gr.block('oblock', prod, 0, 0, 3, 1)

    print gr
    return gr


def make_jet():
    gr = graph.OpGraph()
    gr.vector('eps_dddot', 6)

    gr.se3('T_world_from_body')
    gr.time_antiderivative('eps_ddot', 'eps_dddot')
    gr.time_antiderivative('eps_dot', 'eps_ddot')
    gr.time_antiderivative('T_world_from_body', 'eps_dot')

    return gr


def main():
    imu_observation_model()
    exit(0)

    jet_graph = make_jet()
    print jet_graph
    rk4 = integration.rk4_integrate_no_control(jet_graph)

    cg = CodeGraph(name='integrator', namespaces=['estimation', 'jet'])
    graph_to_cc.express(cg, rk4)

    root = '/home/jacob/repos/experiments/'
    loc = 'estimation/jet'
    name = 'jet_rk4'

    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    main()
