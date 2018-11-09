from collections import defaultdict


def _debug(depth, *text):
    do = False
    if do:
        tt = ' '.join(text)
        spaces = '  ' * depth
        print spaces + tt


def maybe_insert_deps(adj, visited, key, output, depth=0):
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
    ''' TODO: Fix this.'''
    visited = defaultdict(lambda: False)
    result = []
    for element in adj.keys():
        maybe_insert_deps(adj, visited, element, result)
    return result


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
