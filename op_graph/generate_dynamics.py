from text import clang_fmt

import create
import generate
from graph import get_states
from graph_to_cc import to_cpp_type, to_const_ref


def get_controls(gr):
    return gr.to_optimize


def make_types(gr):
    """Generate the state and control structs

    Everything that is related to something else by a derivative is in the state
    Everything that has no children is an input
    Every input that is optimized is a control
    Everything else is an auxiliary quantity
    """

    states = []
    create.create_lvalue('double', 'number0')

    for state in get_states(gr):
        props = gr.properties[state]
        state_lvalue = create.create_lvalue(to_cpp_type(props), state)
        states.append(state_lvalue)

    state_struct = create.create_struct(
        'State', states
    )

    control_lvalues = []
    for control in get_controls(gr):
        props = gr.properties[control]
        u_lvalue = create.create_lvalue(to_cpp_type(props), control)
        control_lvalues.append(u_lvalue)
    control_struct = create.create_struct('Control', control_lvalues)

    print clang_fmt(generate.generate(state_struct))
    print clang_fmt(generate.generate(control_struct))


def generate_dynamics(gr):
    #
    # Generate an integrator
    #
    state_type = create.create_lvalue('State', 'x')
    arguments = [
        to_const_ref(state_type),
        to_const_ref(create.create_lvalue('Control', 'u')),
        create.create_lvalue('double', 'dt'),
    ]
    result = state_type

    states = get_states(gr)

    cb = generate.CodeBlock()

    cb.line('State', 'xn')

    # TODO: "Express" function for any compute graph

    for state in states:
        generator = gr.adj[state]
        if gr.properties[state]['type'] == 'liegroup':
            cb.write("xn.{a} = {subtype}::exp(dt * {dx}) * x.{a};".format(
                a=state,
                subtype=gr.properties[state]['subtype'],
                dx=generator[1][0]
            ))
        else:
            cb.write("xn.{a} = x.{a} + (dt * {dx});".format(
                a=state,
                dx=generator[1][0]
            ))
    cb.line('return', 'xn')
    func = create.create_function('discrete_dynamics', arguments, result, cb)

    print clang_fmt(generate.generate(func))
