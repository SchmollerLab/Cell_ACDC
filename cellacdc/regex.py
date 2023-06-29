def float_regex(allow_negative=True):
    pattern = r'[-+]?[0-9]*\.?[0-9]*[eE]?[\-+]?[0-9]+'
    if not allow_negative:
        pattern.replace('[-+]?', '[+]?')
    return pattern