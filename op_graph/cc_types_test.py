import unittest

import cc_types


class TestCCTypes(unittest.TestCase):
    def test_vector_simple(self):
        vec = 'VecNd<4>'
        vtype = cc_types.typen(vec)
        self.assertEqual(vtype['name'], 'VecNd')
        self.assertFalse(vtype['ref'])
        self.assertFalse(vtype['ptr'])
        self.assertEqual(vtype['template_args'], ['4'])

    def test_vector_ref(self):
        vec = 'VecNd<4>&'
        vtype_ref = cc_types.typen(vec)
        self.assertEqual(vtype_ref['name'], 'VecNd')
        self.assertTrue(vtype_ref['ref'])
        self.assertFalse(vtype_ref['ptr'])

        vec = 'VecNd<4>*'
        vtype_ptr = cc_types.typen(vec)
        self.assertEqual(vtype_ptr['name'], 'VecNd')
        self.assertFalse(vtype_ptr['ref'])
        self.assertTrue(vtype_ptr['ptr'])

        self.assertEqual(len(vtype_ptr['deps']), 1)
        self.assertEqual(vtype_ptr['deps'][0]['header'], 'eigen')

        vec = 'const VecNd<4>*'
        vtype_ptr = cc_types.typen(vec)
        self.assertEqual(vtype_ptr['name'], 'VecNd')
        self.assertFalse(vtype_ptr['ref'])
        self.assertTrue(vtype_ptr['ptr'])
        self.assertIn('const', vtype_ptr['qualifiers'])

        self.assertEqual(len(vtype_ptr['deps']), 1)
        self.assertEqual(vtype_ptr['deps'][0]['header'], 'eigen')

    def test_trivial_type(self):
        name = 'const Sand&'
        _type = cc_types.typen(name)

        self.assertEqual(_type['name'], 'Sand')
        self.assertEqual(len(_type['deps']), 0)
        self.assertTrue(_type['ref'])
        self.assertEqual(_type['qualifiers'], ['const'])

    def test_super_trivial_type(self):
        name = 'Sand'
        _type = cc_types.typen(name)

        self.assertEqual(_type['name'], 'Sand')
        self.assertEqual(len(_type['deps']), 0)
        self.assertFalse(_type['ref'])
        self.assertEqual(_type['qualifiers'], [])


if __name__ == '__main__':
    unittest.main(verbosity=3)
