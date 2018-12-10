import parse
from copy import deepcopy

language_qualifiers = [
    'const',
    'constexpr',
    'volatile',
]


def typen(text):
    """TODO: Use clang."""
    qualifiers = []
    for qualifier in language_qualifiers:
        text, present = parse.get_remove(text, qualifier)

        if present:
            qualifiers.append(qualifier)

    if ('<' in text) and ('>' in text):
        template = parse.between(text, '<', '>')
        text = parse.not_between(text, '<', '>')
        assert ' ' not in text, "{} has a space in it".format(text)

        template_arguments = parse.clean_split(template, ',')
    else:
        template_arguments = []

    # Name
    type_name, ref = parse.get_remove(text, '&')
    if not ref:
        type_name, ptr = parse.get_remove(text, '*')
    else:
        ptr = False

    # Type manipulation
    return {
        'name': type_name,
        'template_args': template_arguments,
        'qualifiers': qualifiers,
        'ref': ref,
        'ptr': ptr,
        # 'deps': type_dep(type_name)
    }


def type_to_str(type_data):
    template = ""
    if len(type_data['template_args']) > 0:
        template = "<{}>".format(", ".join(type_data['template_args']))

    qualifiers = " ".join(type_data['qualifiers'])
    postpenders = {
        'ref': '&',
        'ptr': '*'
    }

    postpend = ""
    for notion, token in postpenders.items():
        if type_data[notion]:
            postpend += token

    end_str = "{qualifiers} {type_name}{template}{postpender}".format(
        qualifiers=qualifiers,
        type_name=type_data['name'],
        template=template,
        postpender=postpend
    )

    return end_str


def header_dep(name):
    return [
        {'header': name}
    ]


def sys_header_dep(name):
    return [
        {'header': name}
    ]


def zero(data):
    ddata = deepcopy(data)
    ddata['ptr'] = False
    ddata['ref'] = False
    ddata['qualifiers'] = []
    full_name = type_to_str(ddata)
    router = {
        'SO3': "SO3()",
        'SE3': "SE3()",
        'SO2': "SO2()",
        'SE2': "SE2()",
        'VecNd': "{}::Zero()".format(full_name),
        'MatNd': "{}::Zero()".format(full_name),
        'double': "0.0",
        'float': "0.0",
    }
    return router.get(data['name'])


def needed_header(type_name):
    mapping = {
        'SO3': header_dep('sophus.hh'),
        'SE3': header_dep('sophus.hh'),
        'SO2': header_dep('sophus.hh'),
        'SE2': header_dep('sophus.hh'),
        'VecNd': header_dep('eigen.hh'),
        'MatNd': header_dep('eigen.hh'),
        'vector': sys_header_dep('vector'),
        'array': sys_header_dep('array'),
    }
    depends = mapping.get(type_name, [])
    return depends


def needed_header_fnc(fnc_name):
    mapping = {
        'dynamic_numerical_gradient': header_dep('numerics/numdiff.hh'),
        'numerical_hessian': header_dep('numerics/num_hessian.hh'),
    }
    depends = mapping.get(fnc_name, [])
    return depends


def type_dep(type_name):
    return type_name
