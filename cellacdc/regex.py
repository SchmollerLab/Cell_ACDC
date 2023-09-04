def float_regex(allow_negative=True):
    pattern = r'[-+]?[0-9]*\.?[0-9]*[eE]?[\-+]?[0-9]+'
    if not allow_negative:
        pattern.replace('[-+]?', '[+]?')
    return pattern

if __name__ == '__main__':
    import re
    s = '0.5, 2.5'
    expr = fr'{float_regex()}'
    m = re.findall(expr, s.replace(' ', ''))
    print(m)