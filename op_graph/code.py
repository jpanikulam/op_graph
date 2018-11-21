from collections import defaultdict

from generate import generate
import create


class CodeGraph(object):
    def __init__(self):
        self._children = []

    def add_child(self, thing, expose=True, inline=True):
        self._children.append(thing)
        deps = {}

    def generate(self):
        text = ""
        for child in self._children:
            text += generate(child)
        return text

    def __str__(self):
        return self.generate()


def test():
    mystruct = create.create_struct(
        'TestStruct',
        [
            create.create_lvalue('double', 'number0'),
            create.create_lvalue('double', 'number1')
        ]
    )

    cg = CodeGraph()
    cg.add_child(mystruct)

    myfunc = create.create_function(
        'perforate',
        [
            create.create_lvalue('double', 'goomba'),
            create.create_lvalue('Fromp&', 'goomba2'),
        ],
        create.create_type('double')
    )

    cg.add_child(myfunc)

    print cg


if __name__ == '__main__':
    test()
