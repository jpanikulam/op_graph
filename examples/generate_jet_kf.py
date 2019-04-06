from op_graph import graph
from op_graph import graph_to_cc
from op_graph import integration
from op_graph.code import CodeGraph
from op_graph import groups
from op_graph import op_defs


def fiducial_observation_model(grx):
    gr = graph.OpGraph()
    gr.copy_types(grx)

    state = gr.emplace('state', gr.group_types['State'])
    parameters = gr.emplace('parameters', gr.group_types['Parameters'])

    # imu_from_vehicle = groups.extract_by_name(gr, 'imu_from_vehicle', parameters, 'T_imu_from_vehicle')

    # world_from_vehicle = camera_from_fiducial

    # camera_from_fiducial =

    gr.register_group_type('FiducialMeasurement', [camera_from_world], [gr.get_properties(camera_from_world)])

    grx.add_graph_as_function(
        'observe_gyro',
        graph=gr,
        output_sym=observed_w,
        input_order=[state, parameters]
    )

    add_error_model(grx, 'FiducialMeasurement', 'observe_fiducial')

    return gr


def gyro_observation_model(grx):
    gr = graph.OpGraph()
    gr.copy_types(grx)

    state = gr.emplace('state', gr.group_types['State'])
    parameters = gr.emplace('parameters', gr.group_types['Parameters'])

    imu_from_vehicle = groups.extract_by_name(gr, 'imu_from_vehicle', parameters, 'T_imu_from_vehicle')
    R_world_from_body = groups.extract_by_name(gr, 'R_world_from_body', state, 'R_world_from_body')

    eps_dot = groups.extract_by_name(gr, 'eps_dot', state, 'eps_dot')
    gyro_bias = groups.extract_by_name(gr, 'gyro_bias', state, 'gyro_bias')
    w_world = gr.block('w_world', eps_dot, 3, 0, 3, 1)

    R_sensor_from_vehicle = gr.rotation('R_sensor_from_vehicle', imu_from_vehicle)

    R_sensor_from_world = gr.mul("R_sensor_from_world", R_sensor_from_vehicle, gr.inv(gr.anon(), R_world_from_body))
    w_imu = gr.mul(gr.anon(), R_sensor_from_world, w_world)

    observed_w = gr.add('observed_w', w_imu, gyro_bias)
    # observed_w = gr.sub('observed_w', gyro_bias, w_imu)

    generated_type = 'GyroMeasurement'
    generated_func = 'observe_gyro'
    gr.register_group_type(generated_type, [observed_w], [gr.get_properties(observed_w)])

    gyro_meas = gr.groupify('gyro_meas', [observed_w], inherent_type=generated_type)

    grx.add_graph_as_function(
        generated_func,
        graph=gr,
        output_sym=gyro_meas,
        input_order=[state, parameters]
    )
    groups.create_group_diff(grx, generated_type)

    add_error_model(grx, generated_type, generated_func)
    return gr


def accel_observation_model(grx):
    gr = graph.OpGraph()
    gr.copy_types(grx)

    generated_type = 'AccelMeasurement'
    generated_func = 'observe_accel'

    gr.register_group_type(
        generated_type,
        ['observed_acceleration'],
        [op_defs.create_vector(3)]
    )

    state = gr.emplace('state', gr.group_types['State'])
    parameters = gr.emplace('parameters', gr.group_types['Parameters'])

    R_world_from_body = groups.extract_by_name(gr, 'R_world_from_body', state, 'R_world_from_body')
    eps_ddot = groups.extract_by_name(gr, 'eps_ddot', state, 'eps_ddot')
    eps_dot = groups.extract_by_name(gr, 'eps_dot', state, 'eps_dot')
    accel_bias = groups.extract_by_name(gr, 'accel_bias', state, 'accel_bias')

    imu_from_vehicle = groups.extract_by_name(gr, 'imu_from_vehicle', parameters, 'T_imu_from_vehicle')

    R_imu_from_vehicle = gr.rotation('R_imu_from_vehicle', imu_from_vehicle)
    w = gr.block('w', eps_dot, 3, 0, 3, 1)
    a_world = gr.block('a_world', eps_ddot, 0, 0, 3, 1)

    R_imu_from_world = gr.mul('R_imu_from_world', R_imu_from_vehicle, gr.inv('R_body_from_world', R_world_from_body))
    a_imu = gr.mul('a_imu', R_imu_from_world, a_world)

    g_world = gr.mul(
        'g_world',
        gr.constant_vector('unit_z', 3, 'unitz'),
        gr.constant_scalar('g_mpss', 9.81)
    )
    g_imu = gr.mul('g_imu', R_imu_from_world, g_world)

    observed_acceleration = gr.reduce_binary_op('add', 'observed_acceleration', [
        a_imu,
        accel_bias,
        g_imu
    ])

    accel_meas = gr.groupify(
        'accel_meas',
        [observed_acceleration],
        inherent_type=generated_type
    )

    grx.add_graph_as_function(
        generated_func,
        graph=gr,
        output_sym=accel_meas,
        input_order=[state, parameters]
    )

    # grx.add_function(
    #     'observe_accel',
    #     returns=grx.group_types[generated_type],
    #     arguments=(
    #         grx.group_types['State'],
    #         grx.group_types['Parameters']
    #     )
    # )

    groups.create_group_diff(grx, generated_type)

    add_error_model(grx, generated_type, generated_func)
    # return grx


def add_error_model(grx, group_type, model_name):
    gr = graph.OpGraph('ErrorModel')
    gr.copy_types(grx)
    gr._copy_functions(grx)

    state = gr.emplace('state', gr.group_types['State'])
    meas = gr.emplace('meas', gr.group_types[group_type])
    parameters = gr.emplace('parameters', gr.group_types['Parameters'])

    expected = gr.func(model_name, 'expected', state, parameters)
    error = gr.func('compute_delta', 'error', meas, expected)

    grx.add_graph_as_function(
        model_name + "_error_model",
        graph=gr,
        output_sym=error,
        input_order=[state, meas, parameters]
    )


def make_jet():
    gr = graph.OpGraph()
    # gr.vector('eps_dddot', 6)
    # gr.vector('daccel_bias', 3)
    # gr.vector('dgyro_bias', 3)
    # gr.time_antiderivative('accel_bias', 'daccel_bias')
    # gr.time_antiderivative('gyro_bias', 'dgyro_bias')
    # gr.time_antiderivative('eps_ddot', 'eps_dddot')

    # gr.vector('g_world', 3)
    gr.se3('T_imu_from_vehicle')

    gr.state(gr.vector('accel_bias', 3))
    gr.state(gr.vector('gyro_bias', 3))
    gr.state(gr.vector('eps_ddot', 6))

    gr.state(gr.time_antiderivative('eps_dot', 'eps_ddot'))

    w = gr.block('w', 'eps_dot', 3, 0, 3, 1)
    v = gr.block('v', 'eps_dot', 0, 0, 3, 1)

    # gr.state(gr.se3('T_body_from_world'))

    # gr.time_antiderivative('T_body_from_world', 'eps_dot')
    gr.state(gr.time_antiderivative('x_world', v))
    gr.so3('R_world_from_body')
    gr.state(gr.time_antiderivative('R_world_from_body', w))
    return gr


def main():

    jet_graph = make_jet()
    rk4 = integration.rk4_integrate_no_control(jet_graph)
    groups.create_group_diff(rk4, 'Parameters')

    accel_observation_model(rk4)
    gyro_observation_model(rk4)

    cg = CodeGraph(name='integrator', namespaces=['estimation', 'jet_filter'])
    graph_to_cc.express(cg, rk4)

    # root = '/home/jacob/repos/experiments/'
    root = '/home/jacob/repos/hover-jet/third_party/experiments/'
    loc = 'estimation/jet'
    name = 'jet_rk4'

    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    main()
