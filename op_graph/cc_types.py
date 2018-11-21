import parse


qualifiers = [
    'const',
    'constexpr',
    'volatile',
]


def typen(text):
    """TODO: Use clang."""
    qualifiers = []
    for qualifier in qualifiers:
        text, present = parse.get_remove(text, qualifier)

        if present:
            qualifiers.append(qualifier)

    if ('<' in text) and ('>' in text):
        template = parse.between(text, '<', '>')
        text = parse.not_between(text, '<', '>')

        template_arguments = parse.clean_split(template, ',')
    else:
        template_arguments = []

    type_name, var_text = text.split(' ')
    # Name
    _, ref = parse.get_remove(var_text, '&')
    _, ptr = parse.get_remove(var_text, '*')

    # Type manipulation
    return {
        'name': type_name,
        'template_args': template_arguments,
        'qualifiers': qualifiers,
        'ref': ref,
        'ptr': ptr,
    }


def header_dep(name):
    return [
        {'header': name}
    ]


def type_dep(type_name):

    mapping = {
        'SO3': header_dep('sophus'),
        'SE3': header_dep('sophus'),
        'SO2': header_dep('sophus'),
        'SE2': header_dep('sophus'),
        'VecNd': header_dep('eigen'),
    }

    depends = mapping.get(type_name, [])
    return depends


def main():
    'VecNd<4>'

    pass

if __name__ == '__main__':
    main()
