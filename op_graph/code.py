import os
from collections import defaultdict

import generate
import create
import cc_types
from text import clang_fmt


class CodeGraph(object):
    def __init__(self, name='my_code', namespaces=[''], force_headers=[]):
        self._name = name

        self._children = []
        self._adj = defaultdict(list)
        self._properties = {}
        self._deps = []
        self._namespaces = namespaces

        self._force_headers = force_headers

    def _get_properties(self, thing):
        return self._properties.get(thing['name'])

    def add_child(self, thing):
        routing = {
            'struct': self.add_struct,
            'function': self.add_function,
        }
        routing[thing['kind']](thing)

    def add_struct(self, struct, expose=True):
        self._children.append(struct)
        self._properties[struct['name']] = {
            'kind': 'struct',
            'expose': expose,
        }

    def add_function(self, func, expose=True):
        self._children.append(func)
        self._properties[func['name']] = {
            'kind': 'func',
            'expose': expose,
        }

    def _recursive_update(self, a, b):
        for k, v in b.items():
            if k in a:
                a[k].update(v)
            else:
                a[k] = v

    def _recurse_namespace(self, namespaces, text):
        scope_name = "namespace {}".format(namespaces[0])
        with generate.Scope(scope_name) as code:
            if len(namespaces) == 1:
                code.write(text)
            else:
                code.write(self._recurse_namespace(namespaces[1:], text))

        return code.code

    def _needed_headers(self, dep_list, expose=False):
        true_deps = {
            'exposed': set(),
            'unexposed': set(),
        }

        list_to_use = 'exposed' if expose else 'unexposed'

        for dep in dep_list:
            if 'deps' in dep.keys():
                if self._get_properties(dep):
                    if self._get_properties(dep)['expose']:
                        sub_expose = True
                    else:
                        sub_expose = False

                new_hdrs = self._needed_headers(dep['deps'], sub_expose)
                self._recursive_update(true_deps, new_hdrs)
            else:
                need_hdr = cc_types.HeaderMapping.needed_header(dep['name'])
                header_names = map(lambda o: o.get('header'), need_hdr)
                true_deps[list_to_use].update(header_names)
        return true_deps

    def generate_source(self, header_name=None):
        if header_name is None:
            header_name = self._name + '.hh'
        hdrs = self._needed_headers(self._children)
        text = "/* Don't edit this; this code was generated by op_graph */\n"

        text += '#include "{}"\n\n'.format(header_name)

        for hdr in hdrs['unexposed']:
            text += '#include "{}"\n'.format(hdr)

        text += "\n"

        code_text = ""
        for child in self._children:
            # props = self._get_properties(child)
            # if props['expose'] and props['kind'] == 'struct':
                # continue
            code_text += generate.generate(child)

        source_text = text + self._recurse_namespace(self._namespaces, code_text)
        return clang_fmt(source_text)

    def generate_header(self):
        text = "/* Don't edit this; this code was generated by op_graph */\n\n"
        text += "#pragma once\n"

        hdrs = self._needed_headers(self._children)
        for hdr in hdrs['exposed']:
            text += '#include "{}"\n'.format(hdr)

        for hdr in self._force_headers:
            text += '#include "{}"\n'.format(hdr)

        text += "\n"

        code_text = ""
        for child in self._children:
            if self._properties[child['name']]['expose']:
                code_text += generate.declare(child)

        header_text = text + self._recurse_namespace(self._namespaces, code_text)
        return clang_fmt(header_text)

    def write_to_files(self, root, subpath, name):
        assert '.' not in name
        paths = os.path.join(subpath, name)

        hdr_path = paths + '.hh'
        with open(os.path.join(root, hdr_path), 'w') as hdr:
            hdr_txt = self.generate_header()
            hdr.write(hdr_txt)

        with open(os.path.join(root, paths + '.cc'), 'w') as cc:
            cc_txt = self.generate_source(hdr_path)
            cc.write(cc_txt)

    def __str__(self):
        return clang_fmt(self.generate_source(self._name + '.hh'))


def test():
    mystruct = create.create_struct(
        'TestStruct',
        [
            create.create_lvalue('double', 'number0'),
            create.create_lvalue('double', 'number1')
        ]
    )

    mystruct2 = create.create_struct(
        'UnexposedTestStruct',
        [
            create.create_lvalue('double', 'number0'),
            create.create_lvalue('SO3', 'number1')
        ]
    )

    cg = CodeGraph(namespaces=['jcc', 'python'])
    cg.add_struct(mystruct)
    cg.add_struct(mystruct2, expose=False)

    myfunc = create.create_function(
        'perforate',
        [
            create.create_lvalue('double', 'goomba'),
            create.create_lvalue('Fromp&', 'goomba2'),
            create.create_lvalue('const VecNd<3>&', 'goomba2'),
        ],
        create.create_type('double')
    )

    cg.add_child(myfunc)

    print '-----Source'
    print cg

    print '-----Header'
    print cg.generate_header()

    root = '/home/jacob/repos'
    loc = 'op_graph/test'
    name = 'test'
    cg.write_to_files(root, loc, name)


if __name__ == '__main__':
    test()
