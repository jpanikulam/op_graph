import os
from subprocess import Popen, PIPE, STDOUT


def form_line(ltokens=[], rtokens=[]):
    """Form line."""
    if not isinstance(ltokens, (list, tuple)):
        return str(ltokens) + ';'

    ltext = ' '.join(map(str, ltokens))
    rtext = ' '.join(map(str, rtokens))
    if len(rtokens) > 0:
        return '{ltext} = {rtext};'.format(ltext=ltext, rtext=rtext)
    else:
        return '{ltext};'.format(ltext=ltext)


def clang_fmt(text, clang_format_path='/home/jacob/repos/llvm/clang-format'):
    """Generate formatted code."""
    if not os.path.exists(clang_format_path):
        raise ValueError("No clang-format!")

    p = Popen([clang_format_path], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    clang_format_stdout = p.communicate(input=str(text))[0]

    text = clang_format_stdout.decode()
    clean_text = text.replace("Can't find usable .clang-format, using LLVM style", "")
    return clean_text
