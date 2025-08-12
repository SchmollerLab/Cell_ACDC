import re

RE_SPLIT_SPACES_IGNORE_QUOTES = re.compile(r'''((?:[^ "']|"[^"]*"|'[^']*')+)''')

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
    return re.sub(r'[^\w\-.]', '_', text)

def get_function_names(text, include_class_methods=True):
    if include_class_methods:
        pattern = r'\bdef\s+([a-zA-Z_]\w*)\s*\('
    else:
        pattern = r'\ndef\s+([a-zA-Z_]\w*)\s*\('
    return re.findall(pattern, text)

def is_alphanumeric_filename(text, allow_space=True):
    if allow_space:
        pattern = r'^[\w\-_. ]+$'
    else:
        pattern = r'^[\w\-_.]+$'
    is_single_or_no_dot = len(re.findall(r'\.', text)) <= 1
    return bool(re.match(pattern, text)) and is_single_or_no_dot

if __name__ == '__main__':
    import re
    s = '0.5, 2.5, nan, NaN'
    expr = fr'{float_regex(include_nan=True)}'
    m = re.findall(expr, s.replace(' ', ''))
    print(m)
    
    s = 'ciao_ciao_-yessa'
    
    print(is_alphanumeric_filename(s))