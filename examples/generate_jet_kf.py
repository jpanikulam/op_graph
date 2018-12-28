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
    gr.copy_types(grx)

    state = gr.emplace('state', gr.group_types['State'])
    parameters = gr.emplace('parameters', gr.group_types['Parameters'])

    sensor_from_vehicle = groups.extract_by_name(gr, 'sensor_from_vehicle', parameters, 'T_sensor_from_body')
    g_world = groups.extract_by_name(gr, 'g_world', parameters, 'g_world')

    vehicle_from_world = groups.extract_by_name(gr, 'vehicle_from_world', state, 'T_body_from_world')
    eps_dot = groups.extract_by_name(gr, 'eps_dot', state, 'eps_dot')
    eps_ddot = groups.extract_by_name(gr, 'eps_ddot', state, 'eps_ddot')
    accel_bias = groups.extract_by_name(gr, 'accel_bias', state, 'accel_bias')

    adj = gr.adjoint('adj', sensor_from_vehicle)

    vw_imu = gr.mul(gr.anon(), adj, eps_dot)
    aq_imu = gr.mul(gr.anon(), adj, eps_ddot)

    v_imu = gr.block('v', vw_imu, 0, 0, 3, 1)
    w_imu = gr.block('w', vw_imu, 3, 0, 3, 1)
    a_imu = gr.block('a', aq_imu, 0, 0, 3, 1)

    clean = gr.sub('clean', gr.cross_product(gr.anon(), w_imu, v_imu), a_imu)

    T_sensor_from_world = gr.mul(gr.anon(), sensor_from_vehicle, vehicle_from_world)
    R_sensor_from_world = gr.rotation('R_sensor_from_world', T_sensor_from_world)
    g_imu = gr.mul('g_imu', R_sensor_from_world, g_world)

    observed_acceleration = gr.reduce_binary_op(
        'add',
        'observed_acceleration',
        [
            clean,
            g_imu,
            accel_bias,
        ]
    )

    gr.register_group_type('AccelMeasurement', [observed_acceleration], [gr.get_properties(observed_acceleration)])

    grx.add_graph_as_function(
        'observe_accel',
        graph=gr,
        output_sym=observed_acceleration,
        input_order=[state, parameters]
    )
    groups.create_group_diff(grx, 'AccelMeasurement')

    return gr


def make_jet():
    gr = graph.OpGraph()
    # gr.vector('eps_dddot', 6)
    # gr.vector('daccel_bias', 3)
    # gr.vector('dgyro_bias', 3)
    # gr.time_antiderivative('accel_bias', 'daccel_bias')
    # gr.time_antiderivative('gyro_bias', 'dgyro_bias')
    # gr.time_antiderivative('eps_ddot', 'eps_dddot')

    gr.vector('g_world', 3)
    gr.se3('T_sensor_from_body')
    gr.se3('T_camera_from_body')

    gr.state(gr.vector('accel_bias', 3))
    gr.state(gr.vector('gyro_bias', 3))
    gr.state(gr.vector('eps_ddot', 6))
    gr.state(gr.time_antiderivative('eps_dot', 'eps_ddot'))
    gr.state(gr.se3('T_body_from_world'))
    gr.time_antiderivative('T_body_from_world', 'eps_dot')
    return gr


def main():

    jet_graph = make_jet()
    rk4 = integration.rk4_integrate_no_control(jet_graph)
    groups.create_group_diff(rk4, 'Parameters')

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
