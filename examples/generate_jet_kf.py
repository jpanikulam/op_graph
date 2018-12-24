from op_graph import graph
from op_graph import graph_to_cc
from op_graph import integration
from op_graph.code import CodeGraph
from op_graph import groups


def gyro_observation_model(grx):
    gr = graph.OpGraph()

    world_from_sensor = gr.se3('T_world_from_sensor')
    eps_dot = gr.vector('eps_dot', 6)

    w = gr.block('w', eps_dot, 3, 0, 3, 1)
    R_world_from_sensor = gr.rotation('R_world_from_sensor', world_from_sensor)

    # observed_w = gr.mul('observed_w', R_world_from_sensor, w)
    observed_w = gr.mul(gr.anon(), gr.inv(gr.anon(), R_world_from_sensor), w)

    grx.add_graph_as_function(
        'observe_gyro',
        graph=gr,
        output_sym=observed_w,
        input_order=[eps_dot]
    )
    return gr


def accel_observation_model(grx):
    gr = graph.OpGraph()
    world_from_sensor = gr.se3('T_world_from_sensor')

    eps_dot = gr.vector('eps_dot', 6)
    eps_ddot = gr.vector('eps_ddot', 6)

    g = gr.vector('gravity_mpss', 3)

    v = gr.block('v', eps_dot, 0, 0, 3, 1)
    w = gr.block('w', eps_dot, 3, 0, 3, 1)

    # a = gr.block('a', eps_ddot, 0, 0, 3, 1)
    # q = gr.block('q', eps_ddot, 3, 0, 3, 1)

    adj = gr.adjoint('adj', world_from_sensor)

    R_world_from_sensor = gr.rotation('R_world_from_sensor', world_from_sensor)

    # R_world_from_sensor = gr.block('R_world_from_sensor', adj, 0, 0, 3, 3)

    Rvx = gr.cross_matrix('Rvx', gr.mul(gr.anon(), R_world_from_sensor, v))
    Rvxw = gr.mul(gr.anon(), Rvx, w)
    vxRw = gr.cross_product(gr.anon(), v, gr.mul(gr.anon(), R_world_from_sensor, w))

    coriolis = gr.add('coriolis', Rvxw, vxRw)
    centrifugal = gr.cross_product('centrifugal', gr.translation(gr.anon(), world_from_sensor), Rvxw)
    inertial_and_euler_all = gr.mul('ad_times_inertial_and_euler', adj, eps_ddot)
    inertial_and_euler = gr.block('inertial_and_euler', inertial_and_euler_all, 0, 0, 3, 1)

    g_imu = gr.mul('g_imu_mpss', gr.inv(gr.anon(), R_world_from_sensor), g)

    observed_acceleration = gr.reduce_binary_op(
        'add',
        'observed_acceleration',
        [
            coriolis,
            centrifugal,
            inertial_and_euler,
            g_imu
        ]
    )
    gr.register_group_type('AccelMeasurement', [observed_acceleration], [gr.get_properties(observed_acceleration)])

    grx.add_graph_as_function(
        'observe_accel',
        graph=gr,
        output_sym=observed_acceleration,
        input_order=[world_from_sensor, eps_dot, eps_ddot, g]
    )
    groups.create_group_diff(grx, 'AccelMeasurement')

    return gr


def make_jet():
    gr = graph.OpGraph()
    gr.vector('eps_dddot', 6)
    gr.vector('daccel_bias', 3)
    gr.vector('dgyro_bias', 3)

    gr.time_antiderivative('accel_bias', 'daccel_bias')
    gr.time_antiderivative('gyro_bias', 'dgyro_bias')

    gr.se3('T_world_from_body')
    gr.time_antiderivative('eps_ddot', 'eps_dddot')
    gr.time_antiderivative('eps_dot', 'eps_ddot')
    gr.time_antiderivative('T_world_from_body', 'eps_dot')

    return gr


def main():

    jet_graph = make_jet()
    rk4 = integration.rk4_integrate_no_control(jet_graph)
    accel_observation_model(rk4)
    gyro_observation_model(rk4)

    cg = CodeGraph(name='integrator', namespaces=['estimation', 'jet_filter'])
    graph_to_cc.express(cg, rk4)

    root = '/home/jacob/repos/experiments/'
    loc = 'estimation/jet'
    name = 'jet_rk4'

    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    main()
