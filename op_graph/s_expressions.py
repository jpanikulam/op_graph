import graph
from log import Log


def apply_s_expression(gr, expr, final):
    if not isinstance(expr, tuple):
        return gr.identity(final, expr)

    args = expr[1]
    new_args = []
    for arg in args:
        if not isinstance(arg, tuple):
            new_args.append(arg)
        else:
            new_sym = apply_s_expression(gr, arg, final=None)
            new_args.append(new_sym)
    op_name = expr[0]

    if final is not None:
        name = final
    else:
        name = gr.anon()
    return gr._call(op_name, name, *new_args)


def apply_binary_simplification(gr, sym):
    op = gr._adj[sym]
    op_type = graph.get_opname(op)
    args = graph.get_args(op)

    types = gr._types(args)
    this_op = gr._op_table[op_type][types]
    if 'zero' in this_op.keys():
        identities = this_op['zero']
        for n, (ident, arg) in enumerate(zip(identities, args)):
            if ident is None:
                continue

            if arg == ident(*args):
                Log.debug("Shortcut: {} <- {}".format(sym, args[n]))
                gr.replace(sym, args[n])

    if 'identity' in this_op.keys():
        identities = this_op['identity']
        for n, (ident, arg) in enumerate(zip(identities, args)):
            if ident is None:
                continue

            if arg == ident(*args):
                Log.debug("Shortcut: {} <- {}".format(sym, args[int(not n)]))
                gr.replace(sym, args[int(not n)])


def scrub_identity(gr, sym):
    op = gr._adj[sym]
    args = graph.get_args(op)
    Log.debug("Shortcut: {} <- {}".format(sym, args[0]))
    gr.replace(sym, args[0])


def scrub_anonymous(gr):
    inv_adj = gr._inverse_adjacency()
    to_remove = []
    for unq in gr.uniques:
        if len(inv_adj[unq]) == 0:
            Log.debug("Removing:", unq)
            to_remove.append(unq)

    for remove in to_remove:
        gr.remove(remove)


def simplify(gr):
    for sym, op in gr._adj.items():
        if op is None:
            continue

        if op[0] in ['mul', 'add']:
            apply_binary_simplification(gr, sym)
        elif op[0] in ['I']:
            scrub_identity(gr, sym)

    scrub_anonymous(gr)
