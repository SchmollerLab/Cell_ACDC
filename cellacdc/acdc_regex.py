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

def is_alphanumeric_filename(
        text, 
        allow_space=True, 
        allowed: str | list[str] | None=None
    ):
    if allow_space:
        pattern = r'^[\w\-_. ]+$'
    else:
        pattern = r'^[\w\-_.]+$'
    
    if allowed is None:
        allowed = []
    
    if isinstance(allowed, str):
        allowed = (allowed,)
    
    max_num_dots = 1
    if allowed is not None:
        max_num_dots += sum([txt.count('.') for txt in allowed])
    
    for allowed_text in allowed:
        allowed_text = re.escape(allowed_text)
        pattern = pattern.replace(r'+$', fr'+({allowed_text})?$')
    
    is_less_max_num_dots = len(re.findall(r'\.', text)) <= max_num_dots
    return bool(re.match(pattern, text)) and is_less_max_num_dots

def get_non_alphanumeric_characters(text):
    return re.findall(r'[^\w\-.]', text)
    
if __name__ == '__main__':
    import re
    s = '0.5, 2.5, nan, NaN'
    expr = fr'{float_regex(include_nan=True)}'
    m = re.findall(expr, s.replace(' ', ''))
    print(m)
    
    s = 'ciao_ciao_-yessa'
    
    print(is_alphanumeric_filename(s))