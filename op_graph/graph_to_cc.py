import generate
import create
from code import CodeGraph
import graph
import graph_tools
import groups
from log import Log

from functools import partial
from copy import deepcopy
import string


def matrix_txt(props):
    if props['dim'][1] == 1:
        return 'VecNd<{}>'.format(props['dim'][0])
    else:
        return 'MatNd<{}, {}>'.format(*props['dim'])


def to_cpp_type(properties):
    choices = {
        'liegroup': lambda: properties['subtype'],
        'matrix': lambda: matrix_txt(properties),
        'scalar': lambda: 'double',
        'group': lambda: properties['inherent_type']
    }
    return choices[properties['type']]()


def to_arg_ref(my_type):
    new_type = deepcopy(my_type)

    primitives = [
        "bool",
        "char",
        "int",
        "size_t",
        "wchar_t",
        "float",
        "double",
    ]

    new_type['type']['qualifiers'].append('const')
    if new_type['type']['name'] in primitives:
        return new_type

    new_type['type']['ref'] = True

    return new_type


def illegal(sym_name):
    illegal = []
    for ill in illegal:
        if ill in sym_name:
            return True
    for char in sym_name:
        if char in string.punctuation.replace('_', ''):
            return True
        if char.isspace():
            return True
    if sym_name[0].isdigit():
        return True
    if sym_name[0] in ['_']:
        return True
    return False


def legalize_it(sym_name):
    repl = {k: '_' for k in string.punctuation}
    new_name = ""
    for char in sym_name:
        new_name += repl.get(char, char)
    return new_name


def generate_constant(sym, gr):
    props = gr.get_properties(sym)

    if props['type'] == 'matrix':
        member_map = {
            'zeros': 'Zero',
            'ones': 'Ones',
            'identity': 'Identity',
            'zero': 'Zero',
        }
        return "{}::{}()".format(matrix_txt(props), member_map[sym.value])
    elif props['type'] == 'scalar':
        value_map = {
            'identity': "1.0",
            'zero': "0.0",
        }
        return value_map.get(sym.value, sym.value)

    raise NotImplementedError("C++ constants not yet implemented for {}".format(props['type']))


def sym_to_text(sym, gr):
    if sym in gr.uniques:
        return '({})'.format(sym_children_to_cc(sym, gr))
    elif gr.is_constant(sym):
        return generate_constant(sym, gr)
    else:
        return str(sym)


def binary(sym, gr, operator=None):
    assert operator is not None
    op = gr.adj[sym]
    args = graph.get_args(op)
    return "{a} {operator} {b}".format(
        a=sym_to_text(args[0], gr),
        b=sym_to_text(args[1], gr),
        operator=operator,
    )


def func(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    func_name = op[0]
    prefix = ""
    if func_name in gr.subgraph_functions:
        overload = gr.get_subgraph_overload(func_name, args)
        if overload['member'] is not None:
            prefix = "{}::".format(overload['member'])

    return prefix + "{}({})".format(
        func_name,
        ','.join(map(partial(sym_to_text, gr=gr), args))
    )


def log(sym, gr):
    args = graph.get_args(gr.adj[sym])
    input_type = gr.get_properties(args[0])['subtype']
    return "{}::log({})".format(input_type, sym_to_text(args[0], gr=gr))


def exp(sym, gr):
    args = graph.get_args(gr.adj[sym])
    result_type = gr.get_properties(sym)['subtype']
    return "{}::exp({})".format(result_type, sym_to_text(args[0], gr=gr))


def hat(sym, gr):
    args = graph.get_args(gr.adj[sym])
    dim = gr.get_properties(sym)['dim']
    mappings = {
        (3, 3): 'SO3',
        (4, 4): 'SE3',
    }
    return "{}::hat({})".format(mappings[dim], sym_to_text(args[0], gr=gr))


def extract(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    assert isinstance(args[1], int), "Oh shit what are you doing"
    struct_name = args[0]
    ind = args[1]
    field_name = gr.get_properties(struct_name)['names'][ind]
    return "{}.{}".format(struct_name, field_name)


def build_struct(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    props = gr.get_properties(sym)

    if props['inherent_type'] is None:
        Log.warn("{} lacks inherent type".format(sym))
    assert props['inherent_type'] is not None, "Can't create type for a group without inherent type."
    new_args = ',\n'.join(map(partial(sym_to_text, gr=gr), args))
    return props['inherent_type'] + "{" + new_args + "}\n"


def inv(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    arg_type = gr.get_properties(args[0])['type']

    if arg_type in ['liegroup', 'matrix']:
        return "{}.inverse()".format(sym_to_text(args[0], gr))
    elif arg_type == 'scalar':
        return "(1.0 / {})".format(sym_to_text(args[0], gr))
    else:
        assert False, "Unupported inverse type"


def vector_index(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    vec_name = args[0]
    index = args[1]
    return "{}[{}]".format(sym_to_text(vec_name, gr), index)


def vstack(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)

    n_args = gr.get_properties(sym)['dim'][0]

    s2t = partial(sym_to_text, gr=gr)
    formed_args = ", ".join(map(s2t, args))
    return "(VecNd<{n}>() << {args}).finished()".format(
        n=n_args,
        args=formed_args
    )


def member(sym, gr, name=None):
    assert name is not None
    args = graph.get_args(gr.adj[sym])

    s2t = partial(sym_to_text, gr=gr)
    formed_args = ", ".join(map(s2t, args[1:]))
    return "{}.{}({})".format(sym_to_text(args[0], gr), name, formed_args)


def block(sym, gr, name=None):
    op = gr.adj[sym]
    args = graph.get_args(op)
    use_sym = sym_to_text(args[0], gr)

    x, y = args[1], args[2],
    x_size, y_size = args[3], args[4],
    return "{}.block<{}, {}>({}, {})".format(use_sym, x_size, y_size, x, y)


def identity(sym, gr):
    args = graph.get_args(gr.adj[sym])
    return sym_to_text(args[0], gr)


def sym_children_to_cc(sym, gr):
    dispatch = {
        'add': partial(binary, operator='+'),
        'sub': partial(binary, operator='-'),
        'mul': partial(binary, operator='*'),
        'div': partial(binary, operator='/'),
        'exp': exp,
        'log': log,
        'hat': hat,
        'groupify': build_struct,
        'extract': extract,
        'inv': inv,
        'adjoint': partial(member, name='Adj'),
        'cross': partial(member, name='cross'),
        'translation': partial(member, name='translation'),
        'rotation': partial(member, name='so3'),
        'block': block,
        'pull': vector_index,
        'vstack': vstack,
        'I': identity,
    }

    op = gr.adj[sym]
    result = dispatch.get(op[0], func)(sym, gr)
    return result


def create_constexpr_int(name):
    cxpr_type = 'static constexpr int'
    return create.create_lvalue(cxpr_type, name)


def all_elements_vector(grp_props):
    for _type, name in zip(grp_props['elements'], grp_props['names']):
        if _type['type'] != 'matrix' or _type['dim'][1] != 1:
            return False
    else:
        return True


def group_to_struct(gr, grp_props, cg, already_generated):
    inherent_type = grp_props['inherent_type']
    assert inherent_type is not None

    lvalues = []
    for _type, name in zip(grp_props['elements'], grp_props['names']):
        cc_type = to_cpp_type(_type)
        lvalues.append(create.create_lvalue(cc_type, name))

    #
    # Add members
    struct_functions = []
    members = gr.group_members[inherent_type]
    for member in members:
        Log.info("Adding: {} to {}".format(member['name'], inherent_type))
        func = to_cc_function(member['name'], member, cg, already_generated)
        struct_functions.append(func)

    #
    # Add ind/dim features
    default_values = dict()
    if all_elements_vector(grp_props):
        index_so_far = 0
        for _type, name in zip(grp_props['elements'], grp_props['names']):
            cc_type = to_cpp_type(_type)
            ind_name = "{}_ind".format(name)
            dim_name = "{}_dim".format(name)
            lvalues.append(create_constexpr_int(ind_name))
            lvalues.append(create_constexpr_int(dim_name))

            dim = groups.get_dim(_type)
            default_values[ind_name] = index_so_far
            default_values[dim_name] = dim
            index_so_far += dim

    #
    # Cardinality
    cardinality = groups.group_cardinality(grp_props)
    cardinality_type = "static constexpr int"
    cardinality_name = "DIM"
    lvalues.append(create.create_lvalue(cardinality_type, cardinality_name))
    default_values[cardinality_name] = cardinality

    #
    # Build the struct
    mystruct = create.create_struct(
        inherent_type,
        lvalues,
        default_values,
        member_functions=struct_functions
    )
    return mystruct


def graph_to_impl(gr, output):
    cb = generate.CodeBlock()

    gr_sorted = graph_tools.topological_sort(gr.adj)
    for sym in gr_sorted:
        if sym not in gr.adj.keys():
            continue

        if sym in gr.uniques:
            continue

        op = gr.adj[sym]

        if op is not None:
            sym_prop = gr.get_properties(sym)
            cpp_type = to_cpp_type(sym_prop)
            decl = "const {type} {name}".format(
                type=cpp_type,
                name=sym
            )
            cb.set(decl, sym_children_to_cc(sym, gr))

    cb.line('return', sym_to_text(output, gr))

    return cb


def to_cc_function(func_name, graph_func, code_graph, generated, to_expose=[]):
    gr = graph_func['graph']

    for name, subgraphs in gr.subgraph_functions.items():
        for subgraph in subgraphs:
            # Don't `generate` member functions
            if subgraph['member'] is not None:
                continue

            if subgraph['unique_id'] in generated:
                Log.warn("Skipping: {}, already generated".format(name))
                continue
            else:
                Log.warn("Generating: {}".format(name))
                generated.add(subgraph['unique_id'])

            sub_function = to_cc_function(name, subgraph, code_graph, generated, to_expose)
            expose = subgraph['unique_id'] in to_expose
            code_graph.add_function(sub_function, expose=expose)

    impl = graph_to_impl(graph_func['graph'], graph_func['output_name'])
    inputs = graph_func['input_names']
    lvalues = []
    types = map(gr.get_properties, inputs)

    for _type, name in zip(types, inputs):
        cc_type = to_cpp_type(_type)
        lvalue = create.create_lvalue(cc_type, name)
        lvalues.append(to_arg_ref(lvalue))

    fname_map = {
        'mul': "operator*",
        'add': "operator+",
        'sub': "operator-",
        'div': "operator/",
    }

    adapted_func_name = fname_map.get(func_name, func_name)

    myfunc = create.create_function(
        adapted_func_name,
        lvalues,
        create.create_type(to_cpp_type(graph_func['returns'])),
        member_of=graph_func['member'],
        impl=impl,
    )

    return myfunc


def express(cg, gr):
    assert isinstance(cg, CodeGraph)
    assert isinstance(gr, graph.OpGraph)

    already_generated = set()

    structs = gr.group_types
    for struct in structs.values():
        nstruct = group_to_struct(gr, struct, cg, already_generated)
        cg.add_struct(nstruct)

    # to_expose = map(lambda o: o.get('unique_id'), gr.subgraph_functions.values())
    to_expose = set()
    for subfuncs in gr.subgraph_functions.values():
        for subfunc in subfuncs:
            to_expose.add(subfunc['unique_id'])

    for name, subgraphs in gr.subgraph_functions.items():
        for subgraph in subgraphs:
            if subgraph['member'] is not None:
                continue

            if subgraph['unique_id'] in already_generated:
                Log.warn("Skipping: {}, already generated".format(name))
                continue
            else:
                Log.warn("Generating: {}".format(name))
                already_generated.add(subgraph['unique_id'])

            cc_func = to_cc_function(name, subgraph, cg, already_generated, to_expose)
            cg.add_function(cc_func)
