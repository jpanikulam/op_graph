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


def simplify(gr):
    import graph

    for sym, op in gr._adj.items():
        if op is None:
            continue

        args = graph.get_args(op)
        types = gr._types(args)

        if op[0] == 'mul':
            this_mul = gr._op_table['mul'][types]

            if 'zero' in this_mul.keys():
                identities = this_mul['zero']
                for n, (ident, arg) in enumerate(zip(identities, args)):
                    if arg == ident(*args):
                        print "Shortcut {} <- {}".format(sym, args[n])
                        gr.replace(sym, args[n])
                    else:
                        pass

            if 'identity' in this_mul.keys():
                identities = this_mul['identity']
                for n, (ident, arg) in enumerate(zip(identities, args)):
                    if arg == ident(*args):
                        print "Shortcut {} <- {}".format(sym, args[int(not n)])
                        gr.replace(sym, args[int(not n)])
                    else:
                        pass
