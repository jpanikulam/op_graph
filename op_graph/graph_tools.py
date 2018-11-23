from collections import defaultdict
from op_defs import Constant


def _debug(depth, *text):
    do = False
    if do:
        tt = ' '.join(text)
        spaces = '  ' * depth
        print spaces + tt


def maybe_insert_deps(adj, visited, key, output, depth=0):
    if isinstance(key, Constant):
        return

    if key in adj.keys():
        _debug(depth, "We have {}".format(key))
        if adj[key] is not None:
            op_args = adj[key][1]
        else:
            op_args = []

        if visited[key]:
            _debug(depth, "Already Visited {}".format(key))
            return

        visited[key] = True

        for dep in op_args:
            _debug(depth, "Beginning Visit:", dep)
            maybe_insert_deps(adj, visited, dep, output, depth + 1)

    else:
        _debug(depth, 'Visiting Unknown:', key)
        visited[key] = True

    _debug(depth, 'Inserting:', key)
    if key not in output:
        output.append(key)


def topological_sort(adj):
    visited = defaultdict(lambda: False)
    result = []
    for element in adj.keys():
        maybe_insert_deps(adj, visited, element, result)
    return result


def mimic_order(to_sort, mimic_this):
    to_sort_set = set(to_sort)
    assert len(to_sort_set - set(mimic_this)) == 0
    new_list = []
    for item in mimic_this:
        if item in to_sort_set:
            new_list.append(item)
    return new_list


def test_topological_sort():
    test = {
        'a': ('dep', ('b', 'd')),
        'b': ('dep', ('c',)),
        'd': ('dep', ('c',)),
        'e': ('dep', ('d',)),
    }

    assert topological_sort(test) == ['c', 'b', 'd', 'a', 'e']

if __name__ == '__main__':
    test_topological_sort()
