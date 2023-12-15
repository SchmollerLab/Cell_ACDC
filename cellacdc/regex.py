def float_regex(allow_negative=True, left_chars=''):
    pattern = r'[-+]?[0-9]*\.?[0-9]*[eE]?[\-+]?[0-9]+'
    if left_chars:
        pattern = fr'{left_chars}{pattern}'
    if not allow_negative:
        pattern.replace('[-+]?', '[+]?')
    return pattern

if __name__ == '__main__':
    import re
    s = '0.5, 2.5'
    expr = fr'{float_regex()}'
    m = re.findall(expr, s.replace(' ', ''))
    print(m)