import graph


def group_cardinality(group_properties):
    count = 0
    for el in group_properties['elements']:
        count += el.get('algebra_dim', el['dim'])[0]
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

    new_group_name = '{}Delta'.format(group_name)
    grx.register_group_type(new_group_name, new_elements)
    delta = grx.groupify('delta', new_elements, inherent_type=new_group_name)

    to_vec_func = create_group_to_vec(grx, new_group_name)
    compute_delta = grx.func(to_vec_func, 'out_vec', delta)

    gr.add_graph_as_function(
        'compute_delta',
        graph=grx,
        output_sym=compute_delta,
        input_order=[grp_a, grp_b]
    )

    create_apply_delta(gr, group_name, new_group_name, grx.get_properties(compute_delta))


def create_apply_delta(gr, group_name, delta_group, delta_props):
    grx = graph.OpGraph('apply_delta')
    struct = gr.group_types[group_name]
    grx.copy_types(gr)

    from_vec_func = create_vec_to_group(grx, delta_group)

    grp_a = grx.emplace('a', struct)
    delta = grx.emplace('delta', delta_props)
    delta_group = grx.func(from_vec_func, 'grp_b', delta)

    out = grx.add('out', grp_a, delta_group)

    gr.add_graph_as_function(
        'apply_delta',
        graph=grx,
        output_sym=out,
        input_order=[grp_a, delta]
    )


def extract_by_name(gr, out_sym, group, field_name):
    grp_props = gr.get_properties(group)
    names = grp_props['names']
    assert field_name in names, "Field name {} not in {}".format(field_name, names)
    ind = names.index(field_name)
    assert ind != -1
    return gr.extract(out_sym, group, ind)


def create_function_jacobian(gr, func):
    # func =
    """
    """

    gr.add_graph_as_function(
        'compute_delta',
        graph=grx,
        output_sym=compute_delta,
        input_order=[grp_a, grp_b]
    )
