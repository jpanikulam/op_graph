import parse


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
        'deps': type_dep(type_name)
    }


def header_dep(name):
    return [
        {'header': name}
    ]


def sys_header_dep(name):
    return [
        {'header': name}
    ]


def needed_header(type_name):
    mapping = {
        'SO3': header_dep('sophus'),
        'SE3': header_dep('sophus'),
        'SO2': header_dep('sophus'),
        'SE2': header_dep('sophus'),
        'VecNd': header_dep('eigen'),
        'MatNd': header_dep('eigen'),
        'vector': sys_header_dep('vector'),
        'array': sys_header_dep('array'),
    }
    depends = mapping.get(type_name, [])
    return depends


def type_dep(type_name):
    return type_name
