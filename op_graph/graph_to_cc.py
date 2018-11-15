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


def sym_children_to_cc(sym, gr):
    dispatch = {
        'add': partial(binary, operator='+'),
        'mul': partial(binary, operator='*'),
        'exp': func,
        'log': func,
        'groupify': build_struct,
        'extract': extract,
        'inv': inv,
        'I': lambda sym, gr: graph.get_args(gr.adj[sym])[0],
    }

    op = gr.adj[sym]
    result = dispatch.get(op[0], func)(sym, gr)
    return result


def group_to_struct(gr, group_sym):
    props = gr.properties[group_sym]
    inherent_type = props['inherent_type']
    assert inherent_type is not None

    lvalues = []
    for _type, name in zip(props['elements'], names):
        cc_type = to_cpp_type(_type)
        lvalues.append(create.create_lvalue(cc_type, name))


    mystruct = create.create_struct(
        inherent_type,
        lvalues
    )
    return mystruct

    cg = CodeGraph()
    cg.add_child(mystruct)

    # myfunc = create.create_function(
    #     'perforate',
    #     [
    #         create.create_lvalue('double', 'goomba'),
    #         create.create_lvalue('Fromp&', 'goomba2'),
    #     ],
    #     'double'
    # )


def express(gr):

    # for group in

    for name, subgraph in gr.subgraph_functions():
        print name + "()"
        express(subgraph['graph'])
        print '----'

    gr_sorted = graph_tools.topological_sort(gr.adj)
    inputs = graph.get_inputs(gr)
    cb = generate.CodeBlock()

    # structs = gr.groups()
    # for struct in structs:
        # nstruct = group_to_struct(gr, struct, )
        # generate.generate_struct(nstruct)


    for sym in gr_sorted:
        if sym not in gr.adj.keys():
            continue

        if gr.is_constant(sym):
            continue

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

    print cb

def test_graph():
    import example_graphs
    # gr = example_graphs.double_integrator()
    # gr = example_graphs.simple_graph()
    gr = example_graphs.rotary_double_integrator()
    # gr = example_graphs.controlled_vectorspring()

    return gr


def main():
    gr = test_graph()
    rk4_gr = integration.rk4_integrate(gr)
    express(rk4_gr)


if __name__ == '__main__':
    main()
