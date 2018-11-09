def form_line(ltokens=[], rtokens=[]):
    if not isinstance(ltokens, (list, tuple)):
        return str(ltokens) + ';'

    ltext = ' '.join(map(str, ltokens))
    rtext = ' '.join(map(str, rtokens))
    if len(rtokens) > 0:
        return '{ltext} = {rtext};'.format(ltext=ltext, rtext=rtext)
    else:
        return '{ltext};'.format(ltext=ltext)
