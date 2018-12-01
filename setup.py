"""OpGraph describes operation graphs
"""
import setuptools

description = "Opgraphs!"

setuptools.setup(
    name='op_graph',
    version='1.0',
    license='MIT',
    long_description=__doc__,
    url='panikul.am',
    author_email='jpanikul@gmail.com',
    packages=setuptools.find_packages(),
    description=description,
    keywords="code generation symbolic math",
    platforms='any',
    zip_safe=True
)
