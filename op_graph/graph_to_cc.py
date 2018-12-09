import generate
import create
from code import CodeGraph
import graph
import graph_tools
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


def sym_to_text(sym, gr):
    if sym in gr.uniques:
        return '({})'.format(sym_children_to_cc(sym, gr))
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
    return "{}({})".format(op[0], ', '.join(map(partial(sym_to_text, gr=gr), args)))


def log(sym, gr):
    args = graph.get_args(gr.adj[sym])
    input_type = gr.get_properties(args[0])['subtype']
    return "{}::log({})".format(input_type, sym_to_text(args[0], gr=gr))


def exp(sym, gr):
    args = graph.get_args(gr.adj[sym])
    result_type = gr.get_properties(sym)['subtype']
    return "{}::exp({})".format(result_type, sym_to_text(args[0], gr=gr))


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

    if arg_type == 'liegroup':
        return "{}.inverse()".format(sym_to_text(args[0], gr))
    else:
        return "(1.0 / {})".format(sym_to_text(args[0], gr))


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
        'groupify': build_struct,
        'extract': extract,
        'inv': inv,
        'pull': vector_index,
        'vstack': vstack,
        'I': identity,
    }

    op = gr.adj[sym]
    result = dispatch.get(op[0], func)(sym, gr)
    return result


def group_to_struct(grp_props):
    inherent_type = grp_props['inherent_type']
    assert inherent_type is not None

    lvalues = []
    for _type, name in zip(grp_props['elements'], grp_props['names']):
        cc_type = to_cpp_type(_type)
        lvalues.append(create.create_lvalue(cc_type, name))

    mystruct = create.create_struct(
        inherent_type,
        lvalues
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

    cb.line('return', output)

    return cb


def to_cc_function(func_name, graph_func, code_graph):
    gr = graph_func['graph']
    for name, subgraph in gr.subgraph_functions():
        sub_function = to_cc_function(name, subgraph, code_graph)
        code_graph.add_function(sub_function, expose=False)
        # print generate.generate(sub_function)

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
        impl=impl
    )

    return myfunc


def express(cg, gr):
    assert isinstance(cg, CodeGraph)
    assert isinstance(gr, graph.OpGraph)

    structs = gr.group_types
    for struct in structs.values():
        nstruct = group_to_struct(struct)
        cg.add_struct(nstruct)

    for name, subgraph in gr.subgraph_functions():
        cc_func = to_cc_function(name, subgraph, cg)
        cg.add_function(cc_func)

    Log.debug('Source------------')
    Log.debug(cg.generate_source())
    Log.debug('Header------------')
    Log.debug(cg.generate_header())
