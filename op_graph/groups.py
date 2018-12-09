import graph


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
            to_stack.extend(pulled)

    out_vec = grx.vstack('out', to_stack)

    return gr.add_graph_as_function(
        'to_vector',
        graph=grx,
        output_sym=out_vec,
        input_order=[input_group]
    )


def create_vec_to_group(gr, group_name):
    grx = graph.OpGraph('extract_{}'.format(group_name))
    struct = gr.group_types[group_name]
    grx.copy_types(gr)
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

    return gr.add_graph_as_function(
        'from_vector',
        graph=grx,
        output_sym=out_grp,
        input_order=[input_vec]
    )


def create_group_diff(gr, group_name):
    grx = graph.OpGraph('delta_{}'.format(group_name))
    struct = gr.group_types[group_name]
    grx.copy_types(gr)

    grp_a = grx.emplace('a', struct)
    grp_b = grx.emplace('b', struct)

    # TODO(jake): There is a bug that arises when 'difference' is anonymous
    # Which causes no code to be generated for it.
    # It appears to be caused by the way uniques are generated.
    # TODO TODO TODO
    subtraction = grx.sub('difference', grp_a, grp_b)

    new_elements = []
    for n, name in enumerate(struct['names']):
        full_dim = struct['elements'][n]['dim']
        assert full_dim[1] == 1

        element_type = struct['elements'][n]['type']

        error_name = "{}_error".format(name)
        assert error_name not in struct['names']

        extracted = grx.extract(error_name, subtraction, n)
        if element_type == 'liegroup':
            log_group = grx.log(error_name + "_log", extracted)
            new_elements.append(log_group)
        else:
            new_elements.append(extracted)

    grx.register_group_type('StateDelta', new_elements)
    delta = grx.groupify('delta', new_elements, inherent_type='StateDelta')

    to_vec_func = create_group_to_vec(grx, 'StateDelta')
    delta_vec = grx.func(to_vec_func, 'out_vec', delta)

    gr.add_graph_as_function(
        'delta_vec',
        graph=grx,
        output_sym=delta_vec,
        input_order=[grp_a, grp_b]
    )