"""Graph tools for CPY
"""
from functools import partial
from graph_tools import topological_sort, mimic_order
from collections import defaultdict, deque
import s_expressions

from log import Log
import op_defs

#
# Generate types in the "platonic ideal"
#


def get_opname(op):
    if op is not None:
        return op[0]
    else:
        return None


def get_args(op):
    if op is not None:
        return op[1]
    else:
        return []


def get_states(gr):
    states = []
    for name, definition in gr.adj.items():
        if definition is not None:
            if definition[0] == 'time_antiderivative':
                states.append(name)
    return set(states)


def get_inputs(gr):
    inputs = []
    for name, definition in gr.adj.items():
        if definition is None:
            inputs.append(name)
    return set(inputs)


def get_parameters(gr):
    parameters = (get_inputs(gr) - gr.to_optimize) - get_states(gr)
    return parameters


class OpGraph(object):
    def __init__(self, name='OpGraph'):
        self._name = name

        # The meat of the thing
        self._adj = {}
        self._properties = {}

        # Functions
        self._functions = defaultdict(list)
        self._subgraph_functions = defaultdict(list)

        # What we're optimizing
        self._to_optimize = set()
        self._outputs = set()

        # A convenience for naming
        self._uniques = set()

        self._group_types = {}

        self._op_table = {}
        self._op_table['mul'] = op_defs.mul(self)
        self._op_table['add'] = op_defs.add(self)
        self._op_table['inv'] = op_defs.inv(self)
        self._op_table['exp'] = op_defs.exp(self)

        self._d_table = {}
        self._d_table['mul'] = op_defs.dmul(self)
        self._d_table['add'] = op_defs.dadd(self)

    def __getattr__(self, key):
        if key in self._op_table.keys():
            return partial(self._call, key)
        else:
            raise AttributeError('No such attribute {}'.format(key))

    @property
    def name(self):
        return self._name

    @property
    def adj(self):
        return self._adj

    @property
    def to_optimize(self):
        return self._to_optimize

    @property
    def uniques(self):
        return self._uniques

    def get_properties(self, sym):
        if isinstance(sym, op_defs.Constant):
            return sym.properties
        return self._properties[sym]

    def _copy_types(self, gr):
        # TODO: Make sure this doesn't overwrite anything
        self._group_types.update(gr._group_types)

    def _copy_functions(self, gr):
        self._subgraph_functions.update(gr._subgraph_functions)
        self._functions.update(gr._functions)

    def mimic_graph(self, gr):
        syms = set(self._adj.keys())
        for sym in gr._to_optimize.intersection(syms):
            self.optimize(sym)

        for sym in gr._uniques.intersection(syms):
            self._uniques.add(sym)

        # Don't always copy functions!
        self._copy_types(gr)

    def subgraph_functions(self):
        for name, funcs in self._subgraph_functions.items():
            for func in funcs:
                yield (name, func)

    def groups(self):
        groups = []
        for sym, prop in self._properties.items():
            if prop['type'] == 'group':
                if prop['elements'] not in groups:
                    groups.append(prop['elements'])
        return groups

    @property
    def group_types(self):
        return self._group_types

    def register_group_type(self, name, field_names=[], field_properties=[]):
        self._needs_all_unique(field_names)

        real_field_props = []
        if len(field_properties) == 0:
            for field_name in field_names:
                self._needs(field_name)
                real_field_props.append(self.get_properties(field_name))
        self._group_types[name] = op_defs.create_group(real_field_props, field_names, inherent_type=name)

    def anon(self):
        return self.unique("anon")

    def unique(self, prefix=None):
        import uuid

        use_prefix = ""
        if prefix is not None:
            use_prefix = prefix + "_"

        new_thing = use_prefix + uuid.uuid4().hex[-6:]
        assert new_thing not in self.adj.keys()
        self._uniques.add(new_thing)
        return new_thing

    #
    # Requirements
    #

    def _default_list(self, default, otherwise):
        self._needs_iter(default)
        self._needs_iter(otherwise)
        if len(otherwise) == 0:
            return default
        return otherwise

    def _type(self, sym):
        self._needs(sym)
        return self.get_properties(sym)['type']

    def _types(self, syms):
        return tuple(map(self._type, syms))

    def _needs(self, sym):
        if isinstance(sym, op_defs.Constant):
            return
        assert sym in self._adj.keys(), "Symbol '{}' does not exist".format(sym)

    def _needs_input(self, sym):
        self._needs(sym)
        assert self._adj[sym] is None, "{} must be an input".format(sym)

    def _needs_not(self, sym):
        """Verify that sym is currently not assigned."""
        if sym in self._adj.keys():
            assert self._adj[sym] is None, "{} must be unset".format(sym)
        else:
            assert sym not in self._adj.keys(), "Symbol '{}' already exists".format(sym)

    def _needs_iter(self, thing):
        assert isinstance(thing, (list, tuple)), "Expected list or tuple"
        return tuple(thing)

    def _needs_inherent_type(self, inherent_type):
        if inherent_type is not None:
            assert inherent_type in self._group_types.keys(), "{} is unknown".format(inherent_type)

    def _inherit(self, from_b):
        if isinstance(from_b, dict):
            return from_b
        else:
            return self.get_properties(from_b)

    def _inherit_first(self, *args):
        assert len(args)
        return self._inherit(args[0])

    def _inherit_last(self, *args):
        assert len(args)
        return self._inherit(args[-1])

    def _needs_same(self, a, b):
        assert self.get_properties(a) == self.get_properties(b), "{} was not {}".format(
            self.get_properties(a),
            self.get_properties(b)
        )

    def _needs_all_unique(self, syms):
        assert len(set(syms)) == len(syms), "All of {} must be unique".format(syms)

    def _needs_properties(self, a, properties):
        assert self.get_properties(a) == properties

    def _needs_same_dim(self, a, b):
        assert self.get_properties(a)['dim'] == self.get_properties(b)['dim']

    def _needs_type(self, a, sym_types):
        if isinstance(sym_types, tuple):
            assert self._type(a) in sym_types
        else:
            assert self._type(a) == sym_types

    def _needs_valid_liegroup(self, kind):
        assert kind in op_defs.VALID_LIEGROUPS

    def _signature_exists(self, name, arguments):
        if name not in self._subgraph_functions.keys():
            return False
        arg_props = tuple(map(self.get_properties, arguments))
        for func in self._subgraph_functions[name]:
            if arg_props == func['args']:
                return True
        else:
            return False

    def func_for_signature(self, name, arguments):
        assert self._signature_exists(name, arguments)
        for func in self._subgraph_functions[name]:
            if arguments == func['args']:
                return func

    def derivative_type(self, sym):
        sym_prop = self.get_properties(sym)

        # It's a lambda so it doesn't evaluate unless dispatched
        outcome_types = {
            'vector': lambda: self._inherit(sym),
            'scalar': lambda: self._inherit(sym),
            'liegroup': lambda: op_defs.create_vector(sym_prop['algebra_dim']),
        }

        return outcome_types[self._type(sym)]()

    def exp_type(self, sym):
        sym_prop = self.get_properties(sym)
        self._needs_type(sym, 'vector')
        assert sym_prop['dim'] in (3, 6)
        if sym_prop['dim'] == 3:
            return op_defs.create_liegroup('SO3')
        else:
            return op_defs.create_liegroup('SE3')

    def _needs_valid_derivative_type(self, sym, dsym):
        """Returns True if dsym can be the derivative of sym."""
        required_derivative_type = self.derivative_type(sym)
        assert self.get_properties(dsym) == required_derivative_type, "{} should have been {}".format(
            self.get_properties(dsym),
            required_derivative_type
        )

    def _copy_subgraph(self, gr, sym, up_to=[], allow_override=False):
        for target in up_to:
            if target not in gr.adj:
                Log.warn("Copying subgraph, but a target: {} is not present ".format(sym))
                assert False

        if sym not in gr.adj:
            return

        if sym in up_to:
            self._adj[sym] = None
            self._properties[sym] = gr.get_properties(sym)
            return

        for o_sym in get_args(gr.adj[sym]):
            self._copy_subgraph(gr, o_sym, up_to=up_to, allow_override=allow_override)

        if not allow_override:
            self._needs_not(sym)

        self._adj[sym] = gr._adj[sym]
        self._properties[sym] = gr.get_properties(sym)

    def insert_subgraph(self, gr, sym, up_to=[]):
        """This allows overriding existing symbols."""
        self._copy_subgraph(gr, sym, up_to=up_to, allow_override=True)

    def extract_subgraph(self, sym, up_to=[]):
        grx = OpGraph('grx')
        grx.insert_subgraph(gr=self, sym=sym, up_to=up_to)
        grx.mimic_graph(self)
        grx._copy_functions(self)
        return grx

    def insert_subgraph_as_function(self, name, gr, output_sym, up_to=[], input_order=[]):
        grx = gr.extract_subgraph(output_sym, up_to=up_to)
        for inp in set(input_order) - set(grx.adj.keys()):
            grx.emplace(inp, gr.get_properties(inp))
        grx._copy_functions(gr)
        self.add_graph_as_function(name, grx, output_sym, input_order=input_order)

    def _inverse_adjacency(self):
        topso = topological_sort(self._adj)

        inv_adjacency = defaultdict(list)
        for sym, op in self._adj.items():
            inv_adjacency[sym]
            for arg in get_args(op):
                inv_adjacency[arg].append(sym)

        for k, v in inv_adjacency.items():
            inv_adjacency[k] = mimic_order(v, topso)

        return inv_adjacency

    def what_depends_on(self, sym):
        """Find out what depends on sym.

        Warning: Constructs the full inverse adjacency on each call.
        """
        return self._inverse_adjacency()[sym]

    def pregroup(self, pregroup_name, syms=[], names=[], inherent_type=None):
        """Group input symbols.

        This is often a convenient operation to apply when creating functions.
        """
        self._needs_iter(syms)
        self._needs_inherent_type(inherent_type)
        self._needs_all_unique(syms)
        self._needs_all_unique(names)
        props = []
        for sym in syms:
            self._needs_input(sym)
            props.append(self.get_properties(sym))

        group_prop = op_defs.create_group(props, names=self._default_list(syms, names), inherent_type=inherent_type)
        self.emplace(pregroup_name, group_prop)
        self.degroupify(syms, pregroup_name)
        return pregroup_name

    #
    # Types
    #

    def is_constant(self, name):
        if isinstance(name, op_defs.Constant):
            return True

        if name in self._adj:
            if self._adj[name] is None:
                return False

            if self._adj[name][0] == 'I':
                args = get_args(self._adj[name])
                return len(args) and not isinstance(args[0], str)
        return False

    def emplace(self, name, properties):
        self._needs_not(name)
        self._adj[name] = None
        self._properties[name] = properties
        return name

    def constant_like(self, mimic_sym, value):
        return op_defs.Constant(self.get_properties(mimic_sym), value)

    def constant_scalar(self, name, value):
        self._adj[name] = self._op('I', value)
        self._properties[name] = op_defs.create_scalar()
        return name

    def scalar(self, name):
        return self.emplace(name, op_defs.create_scalar())

    def vector(self, name, dim):
        return self.emplace(name, op_defs.create_vector(dim))

    def so3(self, name):
        return self.emplace(name, op_defs.create_SO3())

    def se3(self, name):
        return self.emplace(name, op_defs.create_SE3())

    def add_graph_as_function(self, name, graph, output_sym, input_order=[]):
        """Graph can't contain derivatives.

        How to handle groups?
            One idea is to make groups *themeselves* symbols in a style like hcat
        """
        self._needs_iter(input_order)
        self._copy_types(graph)

        returns = graph.get_properties(output_sym)
        graph._needs(output_sym)

        graph_inputs = get_inputs(graph)

        if len(input_order) == 0:
            input_order = tuple(graph_inputs)

        real_input_order = list(input_order)
        for inp in graph_inputs:
            if inp not in input_order:
                real_input_order.append(inp)

        simplified_graph = graph.extract_subgraph(output_sym)
        args = []
        input_map = []
        for inp in real_input_order:
            args.append(graph.get_properties(inp))
            input_map.append(inp)
            simplified_graph.emplace(inp, graph.get_properties(inp))

        args = tuple(args)

        self._subgraph_functions[name].append({
            'graph': simplified_graph,
            'returns': returns,
            'args': args,
            'input_names': tuple(input_map),
            'output_name': output_sym
        })
        self.add_function(name, returns, args)
        return input_map

    def add_function(self, f_name, returns, arguments):
        self._needs_iter(arguments)
        # TODO: Assert no override existing signature
        # assert f_name not in self._functions.keys(), "Function already exists"

        # TODO : use this
        valid_properties = {
            'commutative': (bool, False),
            'associative': (bool, False),
            'positive': (bool, False),
            'negative': (bool, False),
            'invertible': (bool, False),
        }

        self._functions[f_name].append({
            'returns': returns,
            'args': tuple(arguments)
        })
        return f_name

    #
    # Operations (Definition of operations)
    #

    def _op(self, name, *args):
        return (name, args)

    def func(self, func, sym_new, *args):
        assert func in self._functions.keys(), "{} not known".format(func)
        overloaded_funcs = self._functions[func]

        arg_props = tuple([self.get_properties(arg) for arg in args])

        for function in overloaded_funcs:
            if function['args'] == arg_props:
                explicit_func = function
                break
        else:
            Log.warn("No valid function for: {}()".format(func))
            Log.warn(arg_props)
            Log.info("Possible examples: ")
            for function in overloaded_funcs:
                Log.success(function['args'])
            raise KeyError("No valid function.")

        for supplied_arg, expected_property in zip(args, explicit_func['args']):
            self._needs(supplied_arg)
            self._needs_properties(supplied_arg, expected_property)

        self._adj[sym_new] = self._op(func, *args)

        ret_type = explicit_func['returns']
        self._properties[sym_new] = ret_type
        return sym_new

    #
    # Operations (Actual operations)
    #

    def groupify(self, group_sym, syms=[], names=[], inherent_type=None):
        """Creates a group."""
        self._needs_not(group_sym)
        self._needs_inherent_type(inherent_type)
        self._needs_all_unique(syms)
        self._needs_all_unique(names)

        properties = []
        for sym in syms:
            self._needs(sym)
            properties.append(self.get_properties(sym))

        if inherent_type in self._group_types:
            names = self._group_types[inherent_type]['names']

        group_properties = op_defs.create_group(
            properties,
            names=self._default_list(syms, names),
            inherent_type=inherent_type
        )
        self._adj[group_sym] = self._op('groupify', *syms)
        self._properties[group_sym] = group_properties
        return group_sym

    def extend_group(self, new_group_sym, old_group_sym, new_syms=[]):
        self._needs_not(new_group_sym)
        new_properties = []
        old_properties = self.get_properties(old_group_sym)['elements']
        new_properties.extend(old_properties)
        new_properties.extend(map(self.get_properties, new_syms))
        self._adj[new_group_sym] = self._op('extend_group', old_group_sym, *new_syms)
        self._properties[new_group_sym] = op_defs.create_group(new_properties)
        return new_group_sym

    def combine_groups(self, new_group_sym, sym_groups=[]):
        self._needs_not(new_group_sym)
        new_properties = []
        for group in sym_groups:
            old_properties = self.get_properties(group)
            self._needs_type(group, 'group')
            new_properties.extend(old_properties['elements'])

        self._adj[new_group_sym] = self._op('combine_groups', *sym_groups)
        self._properties[new_group_sym] = op_defs.create_group(new_properties)
        return new_group_sym

    def extract(self, sym, group_sym, index):
        self._needs_not(sym)
        props = self.get_properties(group_sym)
        elements = props['elements']
        assert len(elements) > index

        self._adj[sym] = self._op('extract', group_sym, index)
        self._properties[sym] = elements[index]
        return sym

    def degroupify(self, syms, group_sym):
        props = self._properties[group_sym]
        elements = props['elements']
        assert len(syms) == len(elements), "Need same number of symbols as group"
        for n, sym in enumerate(syms):
            self.extract(sym, group_sym, n)
        return syms

    def sym_expand(self, into_gr, sym, expanded):
        raise NotImplementedError()
        op = self._adj[sym]
        if op is None:
            into_gr.emplace(sym, self.self.get_properties(sym))
            expanded[sym]
            return

        if sym in expanded:
            return

        if self._type(sym) == 'group':
            for n, sub_property in enumerate(self.get_properties(sym)['elements']):
                new_sym = self.unique(prefix=sym + "{}".format(n))
                expanded[sym].append(new_sym)
                into_gr.emplace(new_sym, sub_property)
                self.sym_expand(into_gr, sym, expanded)

        if get_opname(op) in ('groupify', 'combine_groups', 'extract'):
            must_define = get_args(op)
            for must_def in must_define:
                print "Must: ", must_def
                if must_def not in expanded:
                    continue
                expanded[must_def] = []
                self.sym_expand(into_gr, must_def, expanded)

    def expand(self):
        """Remove all grouping operations."""
        raise NotImplementedError()
        new_gr = OpGraph("{}Expanded".format(self._name))
        expanded = defaultdict(list)

        for sym in self._adj.keys():
            if sym in expanded:
                continue
            op = self._adj[sym]
            if op is None:
                continue

            self.sym_expand(new_gr, sym, expanded)

        return new_gr

    def _anony_call(self, op_name, *args):
        return self._call(op_name, self.anon(), *args)

    def _broadcast_properties_lat(self, args, n, field):
        subthings = []
        for arg in args:
            if self._type(arg) == 'group':
                subthings.append(self.get_properties(arg)[field][n])
            else:
                subthings.append(self.get_properties(arg))
        return subthings

    def _broadcast_properties_lon(self, arg, field, nargs):
        subthings = []
        if self._type(arg) == 'group':
            subthings = self.get_properties(arg)[field]
        else:
            return tuple()
        return tuple(subthings)

    def _count_args_broadcasted(self, args):
        n_args = None
        for arg in args:
            if (n_args is not None) and (self._type(arg) == 'group'):
                assert len(self.get_properties(arg)['elements']) == n_args
            if self._type(arg) == 'group':
                n_args = len(self.get_properties(arg)['elements'])

        assert n_args is not None
        return n_args

    def _infer_output_group_type(self, outputs, args):
        n_args = self._count_args_broadcasted(args)
        output_props = map(self.get_properties, outputs)
        group_props = map(lambda o: self._broadcast_properties_lon(o, 'elements', n_args), args)

        inherent_type = None
        field_names = []
        if tuple(output_props) in group_props:
            ind = group_props.index(tuple(output_props))
            inherent_type = self.get_properties(args[ind])['inherent_type']
            field_names = self.get_properties(args[ind])['names']

        return field_names, inherent_type

    def _generate_group_func(self, op_name, output_group, *args):
        """Do broadcasted operations.

        If every element of args is a group, this just:
            1. Verifies that an operation exists, op(arg[k][...]) -> X
            2. Creates a function that returns [X] for all k

        If not every element is a group, it repeats the non-group elements and verifies
        that the above operation still exists.
        """
        gr = OpGraph()
        gr.mimic_graph(self)

        for arg in args:
            gr.emplace(arg, self.get_properties(arg))
        n_args = self._count_args_broadcasted(args)

        outputs = []
        for n in range(n_args):
            properties = self._broadcast_properties_lat(args, n, 'elements')
            op_arguments = []
            for k, prop in enumerate(properties):
                if self._type(args[k]) == 'group':
                    element_name = self.get_properties(args[k])['names'][n]
                    op_arguments.append(gr.extract(gr.anon(), args[k], n))
                else:
                    # "Broadcast" if it's not a group
                    element_name = args[k]
                    op_arguments.append(element_name)

            new = gr._call(op_name, gr.anon(), *op_arguments)
            outputs.append(new)

        field_names, inherent_type = gr._infer_output_group_type(outputs, args)
        gr.groupify(output_group, outputs, names=field_names, inherent_type=inherent_type)

        if not self._signature_exists(op_name, args):
            self.add_graph_as_function(op_name, gr, output_group, input_order=args)
        return gr.get_properties(output_group)

    def _call_group(self, op_name, new, *args):
        """Call a function on a group.

        TODO:
            - Eliminate normal _call (I think this can contain it!)
        """
        # new_group_props = self._maybe_generate_group_func(op_name, new, *args)
        new_group_props = self._generate_group_func(op_name, new, *args)

        self._adj[new] = self._op(op_name, *args)
        self._properties[new] = new_group_props

        return new

    def _call(self, op_name, new, *args):
        for arg in args:
            self._needs(arg)

        assert op_name in self._op_table.keys(), "Unknown operation: {}".format(op_name)
        op_def = self._op_table[op_name]

        if 'group' in self._types(args):
            return self._call_group(op_name, new, *args)

        types = tuple(map(lambda o: self._type(o), args))

        assert types in op_def.keys(), "Signature unknown: {} for {}".format(types, op_name)
        real_func = op_def[types]
        self._needs_not(new)

        for need in real_func['needs']:
            need(*args)

        if 'generate' in real_func.keys():
            return real_func['generate'](new, *args)
        else:
            self._adj[new] = self._op(op_name, *args)
            self._properties[new] = real_func['returns'](*args)
        return new

    def simplify(self):
        s_expressions.simplify(self)

    def remove(self, sym):
        self._adj.pop(sym)
        self._properties.pop(sym)
        self._uniques -= {sym}

    def replace(self, from_sym, to_sym):
        """Set sym1 exactly equal to sym2."""
        self._needs_same(from_sym, to_sym)

        if(len(self._inverse_adjacency()[from_sym])) == 0:
            self._adj[from_sym] = self._op('I', to_sym)

        # Think more about guarding against modification during iteration
        while(len(self._inverse_adjacency()[from_sym])):
            ref = self._inverse_adjacency()[from_sym][0]
            old_op = self._adj[ref]
            new_args = list(get_args(old_op))
            ind = new_args.index(from_sym)
            new_args[ind] = to_sym
            self._adj[ref] = self._op(get_opname(old_op), *new_args)

    def forward_mode_differentiate(self, wrt):
        """Differentiate the graph with respect to wrt

        "wrt" -> "w.r.t" -> "with respect to" in case you are a goober
        """
        self._needs_input(wrt)
        inv_adj = self._inverse_adjacency()

        to_diff = deque([wrt])
        diffed = {}
        for inp in get_inputs(self):
            diffed[inp] = self.constant_like(inp, 0)
        diffed[wrt] = self.constant_like(wrt, 1)

        while(len(to_diff)):
            td = to_diff.popleft()
            to_diff.extend(inv_adj[td])
            if td == wrt:
                diffed[td] = op_defs.Constant(self.get_properties(td), 1)
                continue
            if td in diffed.keys():
                continue

            op = self._adj[td]
            if op is not None:
                args = get_args(op)
                arg_types = self._types(args)
                df = self._d_table[get_opname(op)][arg_types]
                df_dx_summands = []
                for n, df_darg in enumerate(df):
                    df_du_sexpr = df_darg['generate'](*args)
                    df_du_sym = self.anon()
                    s_expressions.apply_s_expression(self, df_du_sexpr, df_du_sym)

                    Log.warn("Arg: {}".format(args[n]))
                    Log.warn(diffed)
                    if self.is_constant(args[n]):
                        du_dx = op_defs.Constant(self.get_properties(args[n]), 'zero')
                    else:
                        du_dx = diffed[args[n]]
                    df_dx_summands.append(self._anony_call('mul', df_du_sym, du_dx))

                if td in self._outputs:
                    df_dx_sym = 'd{}_d{}'.format(td, wrt)
                else:
                    df_dx_sym = self.anon()

                if df_dx_sym not in self._adj:
                    total = self.reduce_binary_op('add', df_dx_sym, df_dx_summands)
                else:
                    total = df_dx_sym
                diffed[td] = total

    def reduce_binary_op(self, op, name, args):
        prev = args[0]
        for arg in args[1:-1]:
            prev = self._call(op, self.anon(), prev, arg)
        return self._call(op, name, prev, args[-1])

    def log(self, sym_new, a, kind=None):
        self._needs(a)
        self._needs_valid_liegroup(a)
        self._adj[sym_new] = self._op('log', a)
        a_prop = self.get_properties(a)
        self._properties[sym_new] = op_defs.create_vector(a_prop['algebra_dim'])
        return sym_new

    def time_antiderivative(self, sym, dsym):
        """These are actually a special class of operations.

        In fact, this is a *constraint*. Edges passing through these are permitted to establish cycles
        That's because such cycles are fictitious; they don't exist in the graph unrolled in time.
        """
        self._needs(dsym)

        if sym in self._adj:
            assert sym != dsym
            assert self._adj[sym] is None, "Symbol '{}' already defined!".format(sym)
            self._needs_valid_derivative_type(sym, dsym)
        else:
            self._properties[sym] = self._inherit(dsym)

        self._adj[sym] = self._op('time_antiderivative', dsym)
        return sym

    def identity(self, new_sym, sym):
        self._needs(sym)
        self._needs_not(new_sym)
        self._properties[new_sym] = self.get_properties(sym)
        self._adj[new_sym] = self._op('I', sym)
        return new_sym

    def optimize(self, syms):
        if isinstance(syms, (list, tuple)):
            for sym in syms:
                self._needs(sym)
                self._to_optimize.add(sym)
        else:
            self._needs(syms)
            self._to_optimize.add(syms)

    def output(self, syms):
        if isinstance(syms, (list, tuple)):
            for sym in syms:
                self._needs(sym)
                self._outputs.add(sym)
        else:
            self._needs(syms)
            self._outputs.add(syms)

    #
    # Printing and Visualization
    #

    def warnings(self):
        inv_adjacency = self._inverse_adjacency()
        for sym, parents in inv_adjacency.items():
            if len(parents) == 0:
                Log.warn("[WARN] {} is unused".format(sym))

    def how_do_i_compute(self, sym):
        """Generate the full composed sequence of operations that generates `sym`"""
        self._needs(sym)
        op = self._adj[sym]
        if self.is_constant(sym):
            return sym

        if op is not None:
            sub_ops = tuple(map(self.how_do_i_compute, op[1]))
            replaced_op = (get_opname(op), sub_ops)
        else:
            replaced_op = sym
        return replaced_op

    def print_full_op(self, full_op, depth=0):
        spaces = '  ' * depth
        print spaces + get_opname(full_op), '('
        for thing in full_op[1]:
            if isinstance(thing, tuple):
                self.print_full_op(thing, depth=depth + 1)
            else:
                print spaces + '  ' + thing
        print spaces + ')'

    def symbol_to_text(self, sym, skip_uniques=False):
        if skip_uniques and sym in self._uniques:
            return "({})".format(self.op_to_text(self._adj[sym], skip_uniques=skip_uniques))

        sym_type = self._type(sym)
        sym_prop = self.get_properties(sym)

        if sym_type == 'vector':
            text = "{}[{}]".format(
                sym,
                sym_prop['dim']
            )
        elif sym_type == 'scalar':
                text = "{}".format(sym)
        elif sym_type == 'liegroup':
            group = sym_prop['subtype']
            text = "{}[{}]".format(group, sym)
        elif sym_type == 'group':
            if sym_prop['inherent_type'] is not None:
                gr_type = sym_prop['inherent_type']
            else:
                gr_type = "G"

            text = "{}[{}]".format(gr_type, sym)
        else:
            assert False, "{} unknown".format(sym_type)

        return text

    def op_to_text(self, op, skip_uniques=True):
        sym_to_txt = partial(self.symbol_to_text, skip_uniques=skip_uniques)

        op_name = get_opname(op)
        args = op[1]
        do = {
            'add': lambda o: "{} + {}".format(sym_to_txt(args[0]), sym_to_txt(args[1])),
            'mul': lambda o: "{} * {}".format(sym_to_txt(args[0]), sym_to_txt(args[1])),
            'time_antiderivative': lambda o: "\\int({})".format(sym_to_txt(args[0])),
            'inv': lambda o: "{}^-1".format(sym_to_txt(args[0])),
            'I': lambda o: "{}".format(sym_to_txt(args[0])),
            'null': lambda o: 'null()',
        }

        def alt(operation):
            op_name = operation[0]
            args = operation[1]
            return "{}({})".format(op_name, ', '.join(map(sym_to_txt, args)))

        text = do.get(op_name, alt)(op)
        return text

    def arrows(self, skip_uniques=True):
        """TODO: Make a more useful summary.

        Ideas:
        - Print out the full graph expression as an equation
        - Use tabbing to express dependency depth
            (Some symbols are at different dependency depths for multiple things)
        """
        inv_adj = self._inverse_adjacency()

        total_text = ""
        top_sort = topological_sort(self._adj)
        for sym in top_sort:
            if sym not in self._adj:
                continue
            op = self._adj[sym]

            if op is None:
                total_text += '-> {}\n'.format(self.symbol_to_text(sym, skip_uniques=skip_uniques))
                continue

            # Generally, ignore uniques
            if skip_uniques and (sym in self._uniques):
                # BUT, always print them if nothing depends on them
                if len(inv_adj[sym]) != 0:
                    continue

            text = self.op_to_text(op, skip_uniques=skip_uniques)
            total_text += "{} <- {}\n".format(self.symbol_to_text(sym, skip_uniques=skip_uniques), text)
        return total_text

    def __str__(self):
        return self.arrows()

    def __repr__(self):
        inputs = get_inputs(self)
        types = self._types(inputs)
        things = []
        for input_, type_ in zip(inputs, types):
            things.append("{}: {}".format(type_, input_))

        return "{}({})".format(self._name, ', '.join(things))


def difftest():
    gr = OpGraph('DifferentationGraph')
    gr.scalar('x1')
    gr.scalar('x2')

    a = gr.mul(gr.anon(), 'x1', op_defs.Constant(op_defs.create_scalar(), 'I'))
    # a = gr.mul(gr.anon(), 'x1', 'x1')
    b = gr.mul(gr.anon(), a, 'x1')
    gr.add('c', b, 'x2')
    gr.output('c')

    gr.forward_mode_differentiate('x1')
    print '--------\n\n'
    print gr.arrows(skip_uniques=True)
    gr.simplify()
    print '\n'
    print gr.arrows(skip_uniques=True)

    gr.simplify()
    print gr.arrows(skip_uniques=True)


if __name__ == '__main__':
    difftest()
