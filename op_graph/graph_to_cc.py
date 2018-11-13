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


def to_const_ref(my_type):
    new_type = deepcopy(my_type)
    new_type['type'] = "const {}&".format(my_type['type'])
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


def binary(sym, gr, operator=None):
    assert operator is not None
    op = gr.adj[sym]
    args = graph.get_args(op)
    return "{a} {operator} {b}".format(
        a=args[0],
        b=args[1],
        operator=operator,
    )


def func(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    return "{}({})".format(op[0], ', '.join(map(str, args)))


def build_struct(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    props = gr.properties[sym]
    # return "{}" + "{" + "{}".format(op[0], ', '.join(args))

    if props['inherent_type'] is None:
        Log.warn("{} lacks inherent type".format(sym))
    assert props['inherent_type'] is not None, "Can't create type for a group without inherent type."

    new_args = ',\n'.join(args)

    return props['inherent_type'] + "{" + new_args + "}\n"


def inv(sym, gr):
    op = gr.adj[sym]
    args = graph.get_args(op)
    arg_type = gr.properties[args[0]]
    if arg_type == 'liegroup':
        return "{}.inverse()".format(args[0])
    else:
        return "(1.0 / {})".format(args[0])


def sym_children_to_cc(sym, gr):
    dispatch = {
        'add': partial(binary, operator='+'),
        'mul': partial(binary, operator='*'),
        'exp': func,
        'log': func,
        'groupify': build_struct,
        'inv': inv,
        'I': lambda sym, gr: graph.get_args(gr.adj[sym])[0],
    }

    op = gr.adj[sym]
    # result = dispatch[op[0]](sym, gr)
    result = dispatch.get(op[0], func)(sym, gr)

    return result


def express(gr):
    gr_sorted = graph_tools.topological_sort(gr.adj)
    inputs = graph.get_inputs(gr)
    print 'In:', inputs
    print gr.to_optimize

    print gr.how_do_i_compute('Qn')

    cb = generate.CodeBlock()

    structs = gr.groups()
    for struct in structs:
        print struct

    print gr_sorted
    print gr
    for sym in gr_sorted:
        if gr.is_constant(sym):
            continue
        if sym not in gr.adj.keys():
            continue
        op = gr.adj[sym]
        if op is not None:
            sym_prop = gr.properties[sym]

            # if sym_prop['type'] == 'group':
                # continue

            decl = "const {type} {name}".format(
                type=to_cpp_type(sym_prop),
                name=sym
            )
            cb.set(decl, sym_children_to_cc(sym, gr))

    print cb

def test_graph():
    import example_graphs
    gr = example_graphs.double_integrator()
    # gr = example_graphs.simple_graph()
    # gr = example_graphs.rotary_double_integrator()

    return gr


def main():
    gr = test_graph()
    rk4_gr = integration.rk4_integrate(gr)
    express(rk4_gr)


if __name__ == '__main__':
    main()
