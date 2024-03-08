import re

def float_regex(allow_negative=True, left_chars='', include_nan=False):
    pattern = r'[-+]?[0-9]*\.?[0-9]*[eE]?[\-+]?[0-9]+'
    if left_chars:
        pattern = fr'{left_chars}{pattern}'
    if not allow_negative:
        pattern.replace('[-+]?', '[+]?')
    if include_nan:
        nan_pattern = r'NAN|Nan|NaN|nan'
        pattern = fr'{nan_pattern}|{pattern}'
    return pattern

def to_alphanumeric(text, replacing_char='_'):
    return re.sub('[^\w\-.]', '_', text)

if __name__ == '__main__':
    import re
    s = '0.5, 2.5, nan, NaN'
    expr = fr'{float_regex(include_nan=True)}'
    m = re.findall(expr, s.replace(' ', ''))
    print(m)