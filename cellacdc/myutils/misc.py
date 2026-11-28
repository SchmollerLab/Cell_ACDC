"""Small general-purpose Python helpers, standard library only: text
formatting (salute, trimmed lists and dicts, elided text, ETA, byte
sizes), iteration helpers, chained attributes, safe attribute access and
calling with fewer keyword arguments. translateStrNone and try_kwargs
are plugin API.
"""
import os
import re
import ast
import datetime
from math import pow, ceil


def get_salute_string():
    time_now = datetime.datetime.now().time()
    time_end_morning = datetime.time(12,00,00)
    time_end_lunch = datetime.time(13,00,00)
    time_end_afternoon = datetime.time(15,00,00)
    time_end_evening = datetime.time(20,00,00)
    time_end_night = datetime.time(4,00,00)
    if time_now >= time_end_night and time_now < time_end_morning:
        return 'Have a good day!'
    elif time_now >= time_end_morning and time_now < time_end_lunch:
        return 'Enjoy your lunch!'
    elif time_now >= time_end_lunch and time_now < time_end_afternoon:
        return 'Have a good afternoon!'
    elif time_now >= time_end_afternoon and time_now < time_end_evening:
        return 'Have a good evening!'
    else:
        return 'Have a good night!'


def is_iterable(item):
     try:
         iter(item)
         return True
     except TypeError as e:
         return False


def get_trimmed_list(li: list, max_num_digits=10):
    if len(li) == 0:
        return '[]'
    
    tom_num_digits = sum([len(str(val)) for val in li])
    
    if tom_num_digits == 0:
        return f"[{', '.join(map(str, li))}]"
    
    avg_num_digits = tom_num_digits/len(li)
    max_num_vals = int(round(max_num_digits/avg_num_digits))

    if tom_num_digits > max_num_digits:
        front_vals = ceil(max_num_vals / 2)
        back_vals = max_num_vals // 2
        
        if front_vals + back_vals >= len(li):
            return f"[{', '.join(map(str, li))}]"

        li = li[:front_vals] + ['...'] + li[len(li) - back_vals:]

    return f"[{', '.join(map(str, li))}]"

def get_trimmed_dict(di: dict, max_num_digits=10):
    di_str = di.copy()
    total_num_digits = sum([len(str(key)) + len(str(val)) for key, val in di.items()])
    avg_num_digits = total_num_digits / len(di)
    max_num_vals = int(round(max_num_digits / avg_num_digits))
    if total_num_digits > max_num_digits:
        keys = list(di_str.keys())
        for key in keys[max_num_vals:-max_num_vals]:
            del di_str[key]
        di_str[keys[max_num_vals]] = "..."
    return f"[{', '.join([f'{key} -> {val}' for key, val in di_str.items()])}]"


def _bytes_to_MB(size_bytes):
    factor = pow(2, -20)
    size_MB = round(size_bytes*factor)
    return size_MB

def _bytes_to_GB(size_bytes):
    factor = pow(2, -30)
    size_GB = round(size_bytes*factor, 2)
    return size_GB

def getMemoryFootprint(files_list):
    required_memory = sum([
        48 if file.endswith('.h5') else os.path.getsize(file)
        for file in files_list
    ])
    return required_memory


def seconds_to_ETA(seconds):
    seconds = round(seconds)
    ETA = datetime.timedelta(seconds=seconds)
    ETA_split = str(ETA).split(':')
    if seconds < 0:
        ETA = '00h:00m:00s'
    elif seconds >= 86400:
        days, hhmmss = str(ETA).split(',')
        h, m, s = hhmmss.split(':')
        ETA = f'{days}, {int(h):02}h:{int(m):02}m:{int(s):02}s'
    else:
        h, m, s = str(ETA).split(':')
        ETA = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
    return ETA


def elided_text(text, max_len=50, elid_idx=None):
    if len(text) <= max_len:
        return text

    if elid_idx is None:
        elid_idx = int(max_len/2)
    if elid_idx >= max_len:
        elid_idx = max_len - 1
    idx1 = elid_idx
    idx2 = elid_idx - max_len
    text = f'{text[:idx1]}...{text[idx2:]}'
    return text


def get_chained_attr(_object, _name):
    for attr in _name.split('.'):
        _object = getattr(_object, attr)
    return _object


def find_missing_integers(lst, max_range=None):
    if max_range is not None:
        max_range = lst[-1]+1
    return [x for x in range(lst[0], max_range) if x not in lst]


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def translateStrNone(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, str):
            if arg.lower() == 'none':
                args[i] = None
            elif arg.lower() == 'true':
                args[i] = True
            elif arg.lower() == 'false':
                args[i] = False
    
    return args


def try_kwargs(func, *args, **kwargs):
    """
    Attempt to call a function with the provided arguments and keyword arguments.
    
    If the function raises a TypeError due to unexpected keyword arguments, 
    those arguments are dynamically removed, and the function is retried. 
    This process continues until the function succeeds or no keyword arguments 
    remain, in which case the exception is re-raised.
    
    Args:
        func (Callable): The function to call.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Tuple[Any, List[str]]: A tuple containing:
            - The result of the function call (or None if it fails).
            - A list of keyword arguments that were removed.
    
    Raises:
        ValueError: If a keyword argument mentioned in the error message 
            is not found in the provided kwargs.
        TypeError: If the function fails with a TypeError after all keyword 
            arguments have been removed.
    """
    
    kwargs = kwargs.copy()  # Create a copy to avoid modifying the original
    removed_kwargs = []
    pattern = r"unexpected keyword argument ['\"](\w+)['\"]"    
    while True:
        try:
            return func(*args, **kwargs), removed_kwargs
        except TypeError as e:
            match = re.search(pattern, str(e))
            if match:
                kwarg_name = match.group(1)
                if kwarg_name in kwargs:
                    del kwargs[kwarg_name]
                    removed_kwargs.append(kwarg_name)
                else:
                    raise ValueError(
                        f"Keyword argument '{kwarg_name}' not found in kwargs."
                    )
            else:
                raise e
            
            if len(kwargs) == 0:
                print(f"Function {func.__name__} failed with TypeError: {e}")
                raise e


def safe_get_or_call(obj, path: str):
    """Safely get nested attributes or call methods with literal args from a string path."""
    expr = ast.parse(path, mode='eval').body

    def _eval(node, current_obj):
        if isinstance(node, ast.Attribute):
            return getattr(_eval(node.value, current_obj), node.attr)
        elif isinstance(node, ast.Call):
            func = _eval(node.func, current_obj)
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        elif isinstance(node, ast.Name):
            # First name in chain is assumed to be from `obj`
            return getattr(current_obj, node.id)
        else:
            raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

    return _eval(expr, obj)
