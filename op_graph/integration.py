import graph
from log import Log


def group_cardinality(group_properties):
    count = 0
    for el in group_properties['elements']:
        count += el['dim'][0]
    return count


def create_group_to_vec(gr, group_name):
    grx = graph.OpGraph('extract_{}'.format(group_name))
    struct = gr.group_types[group_name]

    input_group = grx.emplace('in_grp', struct)

    to_stack = []
    for n, name in enumerate(struct['names']):
        full_dim = struct['elements'][n]['dim']
        assert full_dim[1] == 1
        dim = full_dim[0]
        element_type = struct['elements'][n]['type']

        if element_type == 'scalar':
            pulled = grx.extract(grx.anon(), input_group, n)
            to_stack.append(pulled)
        elif element_type == 'matrix':
            pulled = []
            grp_element = grx.extract(grx.anon(), input_group, n)
            for k in range(dim):
                vec_element = grx.pull(grx.anon(), grp_element, k)
                pulled.append(vec_element)
            # vec = grx.vstack(gr.anon(), pulled)
            to_stack.extend(pulled)

    out_vec = grx.vstack('out', to_stack)

    gr.add_graph_as_function(
        'to_vector',
        graph=grx,
        output_sym=out_vec,
        input_order=[input_group]
    )


def create_vec_to_group_function(gr, group_name):
    grx = graph.OpGraph('extract_{}'.format(group_name))
    struct = gr.group_types[group_name]

    # for _type, name in zip(grp_props['elements'], grp_props['names']):
    # hstack_args = []

    grx.copy_types(gr)
    # out_grp = gr.emplace(group_name.lower(), struct)
    # ex = gr.extract(gr.anon(), in_grp, n)
    # gr.pull(gr.anon(), ex, n)

    input_dim = group_cardinality(struct)
    input_vec = grx.vector('in_vec', input_dim)

    group_elements = []
    count = 0
    for n, name in enumerate(struct['names']):

        full_dim = struct['elements'][n]['dim']
        assert full_dim[1] == 1
        dim = full_dim[0]

        element_type = struct['elements'][n]['type']
        if element_type == 'scalar':
            pulled = grx.pull(grx.anon(), input_vec, count)
            group_elements.append(pulled)
        elif element_type == 'matrix':
            pulled = []
            for k in range(count, count + dim):
                vec_element = grx.pull(grx.anon(), input_vec, k)
                pulled.append(vec_element)
            vec = grx.vstack(gr.anon(), pulled)
            group_elements.append(vec)
        else:
            assert False, "Can't extract for a {}".format(element_type)

        count += dim

    out_grp = grx.groupify('out', group_elements, inherent_type=group_name)
    grx.output(out_grp)

    print grx

    gr.add_graph_as_function(
        'from_vector',
        graph=grx,
        output_sym=out_grp,
        input_order=[input_vec]
    )


def insert_qdot_function(gr, rk4):
    states = graph.get_states(gr)

    q = []
    qdot = []
    u = list(gr.to_optimize)
    z = list(graph.get_parameters(gr))

    for state in states:
        statedot = graph.get_args(gr.adj[state])[0]
        qdot.append(statedot)
        q.append(state)

    gr.register_group_type('StateDot', qdot)
    gr.register_group_type('State', q)
    gr.register_group_type('Controls', u)
    gr.register_group_type('Parameters', z)

    Qdot = gr.groupify('Qdot', qdot, inherent_type='StateDot')
    qdot_gr = gr.extract_subgraph(Qdot, up_to=q)

    for qq in q:
        qdot_gr.emplace(qq, gr.get_properties(qq))

    for uu in u:
        qdot_gr.emplace(uu, gr.get_properties(uu))

    for zz in z:
        qdot_gr.emplace(zz, gr.get_properties(zz))

    Q = qdot_gr.pregroup('Q', q, inherent_type='State')
    U = qdot_gr.pregroup('U', u, inherent_type='Controls')
    Z = qdot_gr.pregroup('Z', z, inherent_type='Parameters')

    qdot_gr.mimic_graph(gr)
    rk4.add_graph_as_function(
        'compute_qdot',
        graph=qdot_gr,
        output_sym=Qdot,
        input_order=[Q, U, Z]
    )

    return q, u, z


def rk4_integrate(gr):
    """How do we compute this.

    qdot is an output subgraph of gr
    qdot is a function of some subset of the inputs of gr, including q

    k1 = qdot(q)
    k2 = qdot(q + (h / 2) * k1)
    k3 = qdot(q + (h / 2) * k2)
    k4 = qdot(q + h * k3)

    h2 = h * 2
    dq = (1 / 6) * h2 * (k1 + 2 * (k2 + k3) + k4)
    q1 = q + dq

    qdot: states, controls; side_inputs -> dq_dt
    qdot: x, u; z -> dq_dt

    """
    # Compute qdot
    rk4 = graph.OpGraph('qdot')
    q, u, z = insert_qdot_function(gr, rk4)
    for qq in q:
        rk4.emplace(qq, gr.get_properties(qq))

    for uu in u:
        rk4.emplace(uu, gr.get_properties(uu))

    for zz in z:
        rk4.emplace(zz, gr.get_properties(zz))

    half = rk4.constant_scalar('half', 0.5)
    sixth = rk4.constant_scalar('sixth', 1.0 / 6.0)
    two = rk4.constant_scalar('two', 2.0)
    h = rk4.scalar('h')
    half_h = rk4.mul('half_h', 'h', half)

    Q = rk4.pregroup('Q', q, inherent_type='State')
    U = rk4.pregroup('U', u, inherent_type='Controls')
    Z = rk4.pregroup('Z', z, inherent_type='Parameters')

    K1 = rk4.func('compute_qdot', 'K1', Q, U, Z)

    Q2 = rk4.add('Q2', Q, rk4.mul(rk4.anon(), half_h, K1))
    K2 = rk4.func('compute_qdot', 'K2', Q2, U, Z)

    Q3 = rk4.add('Q3', Q, rk4.mul(rk4.anon(), half_h, K2))
    K3 = rk4.func('compute_qdot', 'K3', Q3, U, Z)

    Q4 = rk4.add('Q4', Q, rk4.mul(rk4.anon(), h, K3))
    K4 = rk4.func('compute_qdot', 'K4', Q4, U, Z)

    k1_and_k4 = rk4.add(rk4.anon(), K1, K4)
    k2_and_k3 = rk4.mul(rk4.anon(), two, rk4.add(rk4.anon(), K2, K3))
    Ksum = rk4.add(rk4.anon(), k1_and_k4, k2_and_k3)

    Qn = rk4.add('Qn', Q, rk4.mul(rk4.anon(), sixth, Ksum))

    for opt in u:
        rk4.optimize(opt)

    rk4_meta = graph.OpGraph('RK4')
    rk4_meta.mimic_graph(gr)
    rk4_meta.add_graph_as_function(
        'rk4_integrate',
        graph=rk4,
        output_sym=Qn,
        input_order=[Q, U, Z, h]
    )

    create_vec_to_group_function(rk4_meta, 'Controls')
    create_group_to_vec(rk4_meta, 'Controls')

    return rk4_meta
