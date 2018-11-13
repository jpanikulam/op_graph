"""Graph tools for CPY
"""
from functools import partial

from graph_tools import topological_sort
from collections import defaultdict
import pprint

from log import Log

MACHINES_CONTROL_EVERYTHING = True

#
# Generate types in the "platonic ideal"
#


def get_args(op):
    if op is not None:
        return op[1]
    else:
        return []


def create_constant(value, properties):
    """TODO is this the right approach?"""
    properties['constant'] = True
    properties['value'] = value
    return properties


def create_scalar():
    prop = {
        'type': 'scalar',
        'dim': 1
    }
    return prop


def create_vector(dim):
    assert isinstance(dim, int)
    prop = {
        'type': 'vector',
        'dim': dim
    }
    return prop


def create_matrix(dim):
    assert isinstance(dim, tuple)
    assert len(dim) == 2
    assert MACHINES_CONTROL_EVERYTHING
    prop = {
        'type': 'matrix',
        'dim': dim
    }
    return prop


def create_liegroup(subtype):
    mapping = {
        'SO3': {'dim': 3, 'algebra_dim': 3},
        'SE3': {'dim': 3, 'algebra_dim': 6},
    }

    default = {
        'type': 'liegroup',
        'subtype': subtype
    }

    default.update(mapping[subtype])
    return default


def create_group(element_properties, inherent_type=None):
    assert isinstance(element_properties, (list, tuple))
    return {
        'type': 'group',
        'elements': tuple(element_properties),
        'inherent_type': inherent_type,
    }


def create_SO3():
    return create_liegroup('SO3')


def create_SE3():
    return create_liegroup('SE3')


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
    _valid_liegroups = [
        'SO3',
        'SE3',
    ]

    def __init__(self, name='OpGraph'):
        self._name = name

        self._adj = {}
        self._properties = {}

        self._functions = defaultdict(list)

        self._subgraph_functions = {}

        self._to_optimize = set()
        self._groups = defaultdict(tuple)
        self._uniques = set()

        self._op_table = {}

        self._op_table['mul'] = self._mul()
        self._op_table['add'] = self._add()
        self._op_table['inv'] = self._inv()
        self._op_table['exp'] = self._exp()

    def __getattr__(self, key):
        if key in self._op_table.keys():
            return partial(self._call, key)

    @property
    def adj(self):
        return self._adj

    @property
    def properties(self):
        return self._properties

    @property
    def to_optimize(self):
        return self._to_optimize

    def groups(self):
        groups = []
        for sym, prop in self._properties.items():
            if prop['type'] == 'group':
                if prop['elements'] not in groups:
                    groups.append(prop['elements'])
        return groups

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

    def _type(self, name):
        self._needs(name)
        return self._properties[name]['type']

    def _types(self, names):
        return tuple(map(self._type, names))

    def _needs(self, name):
        assert name in self._adj.keys(), "Symbol '{}' does not exist".format(name)

    def _needs_input(self, name):
        assert self._adj[name] is None, "{} must be an input".format(name)

    def _needs_not(self, name):
        """Verify that name is currently not assigned."""
        if name in self._adj.keys():
            assert self._adj[name] is None, "{} must be unset".format(name)
        else:
            assert name not in self._adj.keys(), "Symbol '{}' already exists".format(name)

    def _needs_iter(self, thing):
        assert isinstance(thing, (list, tuple)), "Expected list or tuple"
        return tuple(thing)

    def _inherit(self, from_b):
        if isinstance(from_b, dict):
            return from_b
        else:
            return self._properties[from_b]

    def _inherit_first(self, *args):
        assert len(args)
        return self._inherit(args[0])

    def _inherit_last(self, *args):
        assert len(args)
        return self._inherit(args[-1])

    def _needs_same(self, a, b):
        assert self._properties[a] == self._properties[b], "{} was not {}".format(
            self._properties[a],
            self._properties[b]
        )

    def _needs_properties(self, a, properties):
        assert self._properties[a] == properties

    def _needs_same_dim(self, a, b):
        assert self._properties[a]['dim'] == self._properties[b]['dim']

    def _needs_type(self, a, sym_types):
        if isinstance(sym_types, tuple):
            assert self._type(a) in sym_types
        else:
            assert self._type(a) == sym_types

    def _needs_valid_liegroup(self, kind):
        assert kind in self._valid_liegroups

    def derivative_type(self, sym):
        sym_prop = self._properties[sym]

        # It's a lambda so it doesn't evaluate unless dispatched
        outcome_types = {
            'vector': lambda: self._inherit(sym),
            'scalar': lambda: self._inherit(sym),
            'liegroup': lambda: create_vector(sym_prop['algebra_dim']),
        }

        return outcome_types[self._type(sym)]()

    def exp_type(self, sym):
        sym_prop = self._properties[sym]
        self._needs_type(sym, 'vector')
        assert sym_prop['dim'] in (3, 6)
        if sym_prop['dim'] == 3:
            return create_liegroup('SO3')
        else:
            return create_liegroup('SE3')

    def _needs_valid_derivative_type(self, sym, dsym):
        """Returns True if dsym can be the derivative of sym."""
        required_derivative_type = self.derivative_type(sym)
        assert self._properties[dsym] == required_derivative_type, "{} should have been {}".format(
            self.properties[dsym],
            required_derivative_type
        )

    def _copy_subgraph(self, gr, sym, up_to=[], allow_override=False):
        if sym in up_to:
            self._adj[sym] = None
            self._properties[sym] = gr.properties[sym]
            return

        for o_sym in get_args(gr.adj[sym]):
            self._copy_subgraph(gr, o_sym, up_to=up_to, allow_override=allow_override)

        if not allow_override:
            self._needs_not(sym)

        self._adj[sym] = gr._adj[sym]
        self._properties[sym] = gr._properties[sym]

    def insert_subgraph(self, gr, sym, up_to=[]):
        """This allows overriding existing symbols."""
        self._copy_subgraph(gr, sym, up_to, allow_override=True)

    def extract_subgraph(self, sym, up_to=[]):
        grx = OpGraph('grx')
        grx.insert_subgraph(gr=self, sym=sym, up_to=up_to)
        return grx

    def insert_subgraph_as_function(self, name, gr, output_sym, up_to=[], input_order=[]):
        grx = gr.extract_subgraph(output_sym, up_to=up_to)
        for inp in set(input_order) - set(grx.adj.keys()):
            grx.emplace(inp, gr.properties[inp])
        self.add_graph_as_function(name, grx, output_sym, input_order=input_order)

    def _inverse_adjacency(self):
        inv_adjacency = defaultdict(list)
        for sym, op in self._adj.items():
            inv_adjacency[sym]
            for arg in get_args(op):
                inv_adjacency[arg].append(sym)
        return inv_adjacency

    def get_depends_on(self, sym):
        return self._inverse_adjacency()[sym]

    def _one_subs(self, was, should_be):
        def replace(in_tuple, formerly, becomes):
            assert in_tuple.count(formerly) > 0
            # Cheating -- easy to improve
            l_tuple = list(in_tuple)
            while(l_tuple.count(formerly) > 0):
                ind = l_tuple.index(formerly)
                self._needs_same(l_tuple[ind], becomes)
                l_tuple[ind] = becomes
                return tuple(l_tuple)

        self._needs(was)
        self._needs(should_be)
        need_to_change = self.get_depends_on(was)

        # We're flying by the seat of our pants here hoping this is valid!
        # We're just hoping the user doesn't change types
        for target_sym in need_to_change:
            former_op = self._adj[target_sym]
            new_op_args = replace(get_args(former_op), was, should_be)
            self._adj[target_sym] = (former_op[0], new_op_args)

    def subs(self, replacement_dict):
        """This must be adjusted so it doesn't overwrite its own work

        Must be possible to swap in (see notebook)
        NOT DONE"""
        raise NotImplementedError("This function is not implemented, any more questions?")
        for was, should_be in replacement_dict.items():
            self._one_subs(was, should_be)

    def pregroup(self, pregroup_name, syms=[], inherent_type=None):
        self._needs_iter(syms)
        props = []
        for sym in syms:
            self._needs_input(sym)
            props.append(self._properties[sym])

        group_prop = create_group(props, inherent_type=inherent_type)
        self.emplace(pregroup_name, group_prop)
        self.degroupify(syms, pregroup_name)
        return pregroup_name

    #
    # Types
    #

    def is_constant(self, name):
        if name in self._adj:
            if self._adj[name] is None:
                return False

            if self._adj[name][0] == 'identity':
                args = get_args(self._adj[name])
                return len(args) and not isinstance(args[0], str)
        return False

    def emplace(self, name, properties):
        self._adj[name] = None
        self._properties[name] = properties
        return name

    def constant_scalar(self, name, value):
        self._adj[name] = self._op('identity', value)
        self._properties[name] = create_scalar()
        return name

    def scalar(self, name):
        self._adj[name] = None
        self._properties[name] = create_scalar()
        return name

    def vector(self, name, dim):
        self._adj[name] = None
        self._properties[name] = create_vector(dim)
        return name

    def so3(self, name):
        self._adj[name] = None
        self._properties[name] = create_SO3()
        return name

    def se3(self, name):
        self._adj[name] = None
        self._properties[name] = create_SE3()
        return name

    def add_graph_as_function(self, name, graph, output_sym, input_order=[]):
        """Graph can't contain derivatives.

        How to handle groups?
            One idea is to make groups *themeselves* symbols in a style like hcat
        """
        self._needs_iter(input_order)

        returns = graph.properties[output_sym]
        graph._needs(output_sym)

        graph_inputs = get_inputs(graph)

        if len(input_order) == 0:
            input_order = tuple(graph_inputs)

        real_input_order = list(input_order)
        for inp in graph_inputs:
            if inp not in input_order:
                real_input_order.append(inp)

        args = []
        input_map = []
        for inp in real_input_order:
            args.append(graph.properties[inp])
            input_map.append(inp)

        args = tuple(args)

        self._subgraph_functions[name] = {
            'graph': graph,
            'returns': returns,
            'args': args,
            'input_names': tuple(input_map),
        }
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
    # Operations
    #

    def _op(self, name, *args):
        return (name, args)

    def func(self, func, sym_new, *args):
        assert func in self._functions.keys(), "{} not known".format(func)
        overloaded_funcs = self._functions[func]

        arg_props = tuple([self._properties[arg] for arg in args])

        for function in overloaded_funcs:
            if function['args'] == arg_props:
                explicit_func = function
                break
        else:
            Log.warn("No valid function for: {}()".format(
                func))
            Log.warn(pprint.pformat(arg_props, width=50))
            Log.info("Possible examples: ")
            for function in overloaded_funcs:
                Log.success(pprint.pformat(function['args'], width=50), '\n')
            raise KeyError("No valid function.")

        for supplied_arg, expected_property in zip(args, explicit_func['args']):
            self._needs(supplied_arg)
            self._needs_properties(supplied_arg, expected_property)

        self._adj[sym_new] = self._op(func, *args)

        ret_type = explicit_func['returns']
        self._properties[sym_new] = ret_type
        return sym_new

    def groupify(self, group_sym, syms=[], inherent_type=None):
        """Creates a group."""
        # assert len(syms) > 0, "Not enough symbols!"
        self._needs_not(group_sym)

        properties = []
        for sym in syms:
            self._needs(sym)
            properties.append(self._properties[sym])

        group_properties = create_group(properties, inherent_type=inherent_type)
        self._adj[group_sym] = self._op('groupify', *syms)
        self._properties[group_sym] = group_properties
        return group_sym

    def extend_group(self, new_group_sym, old_group_sym, new_syms=[]):
        self._needs_not(new_group_sym)
        # assert len(new_syms) > 0, "Not enough symbols"
        new_properties = []
        old_properties = self._properties[old_group_sym]['elements']
        new_properties.extend(old_properties)
        new_properties.extend(map(self._properties.get(new_syms)))
        self._adj[new_group_sym] = self._op('extend_group', old_group_sym, *new_syms)
        self._properties[new_group_sym] = create_group(new_properties)
        return new_group_sym

    def combine_groups(self, new_group_sym, sym_groups=[]):
        self._needs_not(new_group_sym)
        new_properties = []
        for group in sym_groups:
            old_properties = self._properties[group]
            self._needs_type(group, 'group')
            new_properties.extend(old_properties['elements'])

        self._adj[new_group_sym] = self._op('combine_groups', *sym_groups)
        self._properties[new_group_sym] = create_group(new_properties)
        return new_group_sym

    def extract(self, sym, group_sym, index):
        self._needs_not(sym)
        props = self._properties[group_sym]
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
            into_gr.emplace(sym, self._properties[sym])
            expanded[sym]
            return

        if sym in expanded:
            return

        if self._type(sym) == 'group':
            for n, sub_property in enumerate(self._properties[sym]['elements']):
                new_sym = self.unique(prefix=sym + "{}".format(n))
                expanded[sym].append(new_sym)
                into_gr.emplace(new_sym, sub_property)

                self.sym_expand(into_gr, sym, expanded)

        if op[0] in ('groupify', 'combine_groups', 'extract'):
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
        return self._call(op_name, 'anon_' + self.unique(), *args)

    def _call_group(self, op_name, new, *args):
        """Now, the trick is to represent "implicit graphs" that arise
        from groups.

        Unanswered Questions:
            - What does SO3(R) + x[3] look like, as groups?
            - How do you traverse a graph with a group?
                - Should it be handled for you??
        """
        for arg in args:
            assert self._type(arg) == 'group'

        op_def = self._op_table[op_name]
        group_args = map(lambda o: self._properties[o]['elements'], args)

        outputs = []
        for elements in zip(*group_args):
            types = tuple(map(lambda o: o['type'], elements))
            assert types in op_def.keys(), "Unknown {} for arguments {}".format(op_name, types)
            explicit_func = op_def[types]
            output = explicit_func['returns'](*elements)
            outputs.append(output)

        # if tuple(outputs) == group_args[-1]:
            # inherent_type = self._properties[args[-1]]['inherent_type']
        if tuple(outputs) in group_args:
            ind = group_args.index(tuple(outputs))
            inherent_type = self._properties[args[ind]]['inherent_type']
        else:
            inherent_type = None

        new_group = create_group(outputs, inherent_type)
        self._adj[new] = self._op(op_name, *args)
        self._properties[new] = new_group
        return new

    def _call(self, op_name, new, *args):
        for arg in args:
            self._needs(arg)

        assert op_name in self._op_table.keys(), "Unknown operation: {}".format(op_name)
        op_def = self._op_table[op_name]

        if 'group' in self._types(args):
            return self._call_group(op_name, new, *args)

        types = tuple(map(lambda o: self._type(o), args))

        assert types in op_def.keys(), "Signature unknown: {}".format(types)
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

    def _inv(self):
        return {
            ('liegroup',): {
                'returns': self._inherit_last,
                'needs': [],
            },
            ('scalar',): {
                'returns': self._inherit_last,
                'needs': [],
            }
        }

    def _add(self):
        return {
            ('liegroup', 'vector'): {
                'returns': self._inherit_first,
                'needs': [self._needs_valid_derivative_type],
                'generate': lambda n, a, b: self._call('mul', n, a, self._anony_call('exp', b))
            },
            ('vector', 'vector'): {
                'returns': self._inherit_last,
                'needs': [self._needs_same]
            },
            ('scalar', 'scalar'): {
                'returns': self._inherit_last,
                'needs': [self._needs_same]
            },
        }

    def _mul(self):
        # How do we generate *template* types?
        # Eesh, does this mean we have to define a template system?
        return {
            ('liegroup', 'vector'): {
                'returns': self._inherit_last,
                'needs': [self._needs_same_dim]
            },
            ('liegroup', 'liegroup'): {
                'returns': self._inherit_last,
                'needs': [self._needs_same]
            },
            ('scalar', 'vector'): {
                'returns': self._inherit_last,
                'needs': []
            },
            ('scalar', 'scalar'): {
                'returns': self._inherit_last,
                'needs': [self._needs_same]
            },
        }

    def _exp(self):
        return {
            ('vector',): {
                'returns': self.exp_type,
                'needs': []
            },
            ('scalar',): {
                'returns': lambda a: 'scalar',
                'needs': []
            }
        }

    def log(self, sym_new, a, kind=None):
        self._needs(a)
        self._needs_valid_liegroup(a)
        self._adj[sym_new] = self._op('log', a)
        a_prop = self._properties[a]
        self._properties[sym_new] = create_vector(a_prop['algebra_dim'])
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
        self._properties[new_sym] = self._properties[sym]
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
            replaced_op = (op[0], sub_ops)
        else:
            replaced_op = sym
        return replaced_op

    def print_full_op(self, full_op, depth=0):
        spaces = '  ' * depth
        print spaces + full_op[0], '('
        for thing in full_op[1]:
            if isinstance(thing, tuple):
                self.print_full_op(thing, depth=depth + 1)
            else:
                print spaces + '  ' + thing
        print spaces + ')'

    def to_text(self, sym):
        if sym in self._uniques:
            return "({})".format(self.op_to_text(self._adj[sym]))


        sym_type = self._type(sym)
        sym_prop = self._properties[sym]

        if sym_type == 'vector':
            text = "{}[{}]".format(
                sym,
                self._properties[sym]['dim']
            )
        elif sym_type == 'scalar':
                text = "{}".format(sym)
        elif sym_type == 'liegroup':
            group = sym_prop['subtype']
            text = "{}[{}]".format(group, sym)
        elif sym_type == 'group':
            text = "G[{}]".format(sym)
        else:
            assert False, "{} unknown".format(sym_type)

        return text

    def op_to_text(self, op):
        op_name = op[0]
        args = op[1]
        do = {
            'add': lambda o: "{} + {}".format(self.to_text(args[0]), self.to_text(args[1])),
            'mul': lambda o: "{} * {}".format(self.to_text(args[0]), self.to_text(args[1])),
            'time_antiderivative': lambda o: "\\int({})".format(self.to_text(args[0])),
            'inv': lambda o: "{}^-1".format(self.to_text(args[0])),
            'null': lambda o: 'null()',
        }

        def alt(operation):
            op_name = operation[0]
            args = operation[1]
            return "{}({})".format(op_name, ', '.join(map(str, args)))

        text = do.get(op_name, alt)(op)
        return text

    def arrows(self):
        """TODO: Make a more useful summary.

        Ideas:
        - Print out the full graph expression as an equation
        - Use tabbing to express dependency depth
            (Some symbols are at different dependency depths for multiple things)
        """
        total_text = ""
        top_sort = topological_sort(self._adj)
        for sym in top_sort:
            if sym not in self._adj:
                continue
            op = self._adj[sym]

            if op is None:
                total_text += '-> {}\n'.format(self.to_text(sym))
                continue

            if sym in self._uniques:
                continue

            text = self.op_to_text(op)
            total_text += "{} <- {}\n".format(self.to_text(sym), text)
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


def grouptest():
    a = ['a0', 'a1', 'a2']
    b = ['b0', 'b1', 'b2']
    m = ['m0', 'm1', 'm2']
    s = ['s0', 's1', 's2']

    out = 'Q'

    gr = OpGraph()

    for aa in a:
        gr.vector(aa, 3)

    for bb in b:
        gr.vector(bb, 3)

    for mm in m:
        gr.scalar(mm)

    for ss in s:
        gr.so3(ss)

    A = gr.groupify("A", a)
    B = gr.groupify("B", b)
    M = gr.groupify("M", m)
    S = gr.groupify("S", s)

    print gr.add("C", A, B)
    gr.add("D", "C", B)
    gr.mul("E", M, "D")
    gr.mul("R", S, "D")

    gr.extract('r1', "R", 1)

    print gr


def main():
    gr = OpGraph()

    gr.scalar('super_density')
    gr.scalar('mass')
    gr.so3('R')

    gr.vector('q', 3)

    gr.mul('density', 'mass', 'super_density')
    gr.mul('a', gr.inv('inv_density', 'density'), 'q')
    gr.mul('Ra', 'R', 'a')

    # Broken -- Define op
    gr.add('RRa', 'R', 'Ra')
    gr.time_antiderivative('R', 'q')

    gr.groupify('Out', ['a', 'Ra', 'R'])

    gr2 = OpGraph()
    gr2.insert_subgraph(gr, 'Out', up_to=['inv_density'])

    gr3 = OpGraph()
    gr3.add_graph_as_function('poopy_func', gr2, 'Out')

    gr3.scalar('mass')
    gr3.vector('u', 3)

    gr3.func('poopy_func', 'rxx', 'u', 'mass')
    gr3.constant_scalar('half', 0.5)

    print gr
    print gr2
    print gr3._subgraph_functions
    print gr3

    print gr3.adj['rxx']
    print gr3.properties['rxx']


if __name__ == '__main__':
    main()
    # grouptest()
