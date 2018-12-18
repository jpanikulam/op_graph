from op_graph import graph
from op_graph import graph_to_cc
from op_graph import integration
from op_graph.code import CodeGraph
from op_graph import groups


def imu_observation_model():
    gr = graph.OpGraph()
    gr.vector('eps_dot', 6)


def make_jet():
    gr = graph.OpGraph()
    gr.vector('eps_dddot', 6)

    gr.se3('T_world_from_body')
    gr.time_antiderivative('eps_ddot', 'eps_dddot')
    gr.time_antiderivative('eps_dot', 'eps_ddot')
    gr.time_antiderivative('T_world_from_body', 'eps_dot')

    return gr


def main():
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
