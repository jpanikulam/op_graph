from op_graph import graph
from op_graph import graph_to_cc
from op_graph.code import CodeGraph
from op_graph import groups

from op_graph.cc_types import HeaderMapping, header_dep, sys_header_dep
HeaderMapping.set_header_mapping({
    'SO3': header_dep('third_party/experiments/sophus.hh'),
    'SE3': header_dep('third_party/experiments/sophus.hh'),
    'SO2': header_dep('third_party/experiments/sophus.hh'),
    'SE2': header_dep('third_party/experiments/sophus.hh'),
    'VecNd': header_dep('third_party/experiments/eigen.hh'),
    'MatNd': header_dep('third_party/experiments/eigen.hh'),
    'vector': sys_header_dep('vector'),
    'array': sys_header_dep('array'),
})


def make_vanes():
    gr = graph.OpGraph()

    vane_0 = gr.scalar('servo_0_angle')
    vane_1 = gr.scalar('servo_1_angle')
    vane_2 = gr.scalar('servo_2_angle')
    vane_3 = gr.scalar('servo_3_angle')

    vanes = [vane_0, vane_1, vane_2, vane_3]
    gr.register_group_type('QuadraframeStatus', vanes, gr.get_properties(vanes))
    groups.create_group_diff(gr, 'QuadraframeStatus')
    return gr


def make_wrench():
    gr = graph.OpGraph()
    force = gr.vector('force_N', 3)
    torque = gr.vector('torque_Nm', 3)
    elements = [force, torque]
    gr.register_group_type('Wrench', elements, gr.get_properties(elements))

    wrench = gr.groupify('wrench', elements, inherent_type='Wrench')
    groups.create_group_diff(gr, 'Wrench')
    return gr


def vanes():
    vane_gr = make_vanes()
    exp_headers = ['third_party/experiments/eigen.hh']
    cg = CodeGraph(name='vanes', namespaces=['jet', 'control'], force_headers=exp_headers)
    graph_to_cc.express(cg, vane_gr)
    root = '/home/jacob/repos/hover-jet/'
    loc = 'control'
    name = 'vanes'
    cg.write_to_files(root, loc, name)


def wrench():
    wrench_gr = make_wrench()

    exp_headers = ['third_party/experiments/eigen.hh']
    cg = CodeGraph(name='wrench', namespaces=['jet', 'control'], force_headers=exp_headers)
    graph_to_cc.express(cg, wrench_gr)
    root = '/home/jacob/repos/hover-jet/'
    loc = 'control'
    name = 'wrench_generated'
    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    vanes()
    wrench()
