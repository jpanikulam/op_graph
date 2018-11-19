import generate
import create
import graph
import graph_tools
import integration
from log import Log

from functools import partial
from copy import deepcopy
import string


def to_cpp_type(properties):
    choices = {
        'liegroup': lambda: properties['subtype'],
        'vector': lambda: 'VecNd<{}>'.format(properties['dim']),
        'scalar': lambda: 'double',
        'group': lambda: properties['inherent_type']
    }
    return choices[properties['type']]()


def to_arg_ref(my_type):
    new_type = deepcopy(my_type)

    primitives = [
        "bool",
        "char",
        "short int",
        "int",
        "long",
        "size_t",
        "wchar_t",
        "float",
        "double",
    ]

    if my_type['type']['name'] in primitives:
        return my_type

    new_type['type']['name'] = "const {}&".format(my_type['type']['name'])
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


def exp(sym, gr):
    args = graph.get_args(gr.adj[sym])
    result_type = gr.properties[sym]['subtype']
    return "{}::exp({})".format(result_type, sym_to_text(args[0], gr=gr))


def extract(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    assert isinstance(args[1], int), "Oh shit what are you doing"
    struct_name = args[0]
    ind = args[1]
    field_name = gr.properties[struct_name]['names'][ind]
    return "{}.{}".format(struct_name, field_name)


def build_struct(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    props = gr.properties[sym]

    if props['inherent_type'] is None:
        Log.warn("{} lacks inherent type".format(sym))
    assert props['inherent_type'] is not None, "Can't create type for a group without inherent type."
    new_args = ',\n'.join(map(partial(sym_to_text, gr=gr), args))
    return props['inherent_type'] + "{" + new_args + "}\n"


def inv(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    arg_type = gr.properties[args[0]]
    if arg_type == 'liegroup':
        return "{}.inverse()".format(sym_to_text(args[0], gr))
    else:
        return "(1.0 / {})".format(sym_to_text(args[0], gr))


def identity(sym, gr):
    args = graph.get_args(gr.adj[sym])
    return sym_to_text(args[0], gr)


def sym_children_to_cc(sym, gr):
    dispatch = {
        'add': partial(binary, operator='+'),
        'mul': partial(binary, operator='*'),
        'exp': exp,
        'log': func,
        'groupify': build_struct,
        'extract': extract,
        'inv': inv,
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

        # if gr.is_constant(sym):
            # print 'CONSTANT'
            # continue

        if sym in gr.uniques:
            continue

        op = gr.adj[sym]

        if op is not None:
            sym_prop = gr.properties[sym]
            decl = "const {type} {name}".format(
                type=to_cpp_type(sym_prop),
                name=sym
            )
            cb.set(decl, sym_children_to_cc(sym, gr))

    cb.line('return', output)

    return cb


def to_cc_function(func_name, graph_func):
    gr = graph_func['graph']

    impl = graph_to_impl(graph_func['graph'], graph_func['output_name'])
    inputs = graph_func['input_names']

    lvalues = []
    types = map(gr.properties.get, inputs)

    for _type, name in zip(types, inputs):
        cc_type = to_cpp_type(_type)
        lvalue = create.create_lvalue(cc_type, name)
        lvalues.append(to_arg_ref(lvalue))

    fname_map = {
        'mul': "operator*",
        'add': "operator+",
        'sub': "operator-",
    }

    adapted_func_name = fname_map.get(func_name, func_name)

    myfunc = create.create_function(
        adapted_func_name,
        lvalues,
        create.create_type(to_cpp_type(graph_func['returns'])),
        impl=impl
    )

    return myfunc


def express(gr):
    structs = gr.group_types
    for struct in structs.values():
        nstruct = group_to_struct(struct)
        print generate.generate(nstruct)

    for name, subgraph in gr.subgraph_functions():
        print generate.generate(to_cc_function(name, subgraph))


def test_graph():
    import example_graphs
    # gr = example_graphs.double_integrator()
    gr = example_graphs.simple_graph()
    # gr = example_graphs.rotary_double_integrator()
    # gr = example_graphs.controlled_vectorspring()

    return gr


def main():
    gr = test_graph()
    rk4_gr = integration.rk4_integrate(gr)
    express(rk4_gr)


if __name__ == '__main__':
    main()
