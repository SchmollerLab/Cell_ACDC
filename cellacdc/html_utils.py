from functools import wraps
import re
import sys

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from matplotlib.colors import to_hex

is_mac = sys.platform == 'darwin'

def _tag(tag_info='p style="font-size:10px"'):
    def wrapper(func):
        @wraps(func)
        def inner(text):
            tag = tag_info.split(' ')[0]
            text = f'<{tag_info}>{text}</{tag}>'
            return text
        return inner
    return wrapper

def tag(text, tag_info='p style="font-size:10pt"'):
    tag = tag_info.split(' ')[0]
    text = f'<{tag_info}>{text}</{tag}>'
    return text

def href_tag(text, url):
    txt = tag(text, tag_info=f'a href="{url}"')
    return txt

def to_list(items, ordered=False):
    list_tag = 'ol' if ordered else 'ul'
    items_txt = ''.join([f'<li>{item}</li>' for item in items])
    txt = tag(items_txt, tag_info=list_tag)
    return txt

def span(text, color='r'):
    try:
        c = to_hex(color)
    except Exception as e:
        if color == 'r':
            c = 'red'
        elif color == 'g':
            c = 'green'
        elif color == 'k':
            c = 'black'
        else:
            c = color
    return f'<span style="color: {c}">{text}</span>'

def css_head(txt):
    # if is_mac:
    #     txt = txt.replace(',', ',&nbsp;')
    s = (f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style type="text/css">
            {txt}
    </style>
    </head>
    """)
    return s

def html_body(txt):
    if is_mac:
        txt = txt.replace(',', ',&nbsp;')
    s = (f"""
    <body>
        {txt}
    </body>
    </html>
    """)
    return s

def paragraph(txt, font_size='13px', font_color=None, wrap=True, center=False):
    # if is_mac:
    #     # Qt < 5.15.3 has a bug on macOS and the space after comma and perdiod
    #     # are super small. Force a non-breaking space (except for 'e.g.,').
    #     txt = txt.replace(',', ',&nbsp;')
    #     txt = txt.replace('.', '.&nbsp;')
    #     txt = txt.replace('e.&nbsp;g.&nbsp;', 'e.g.')
    #     txt = txt.replace('.&nbsp;.&nbsp;.&nbsp;', '...')
    #     txt = txt.replace('i.&nbsp;e.&nbsp;', 'i.e.')
    #     txt = txt.replace('etc.&nbsp;)', 'etc.)')
    if not wrap:
        txt = txt.replace(' ', '&nbsp;')
    if font_color is None:
        s = (f"""
        <p style="font-size:{font_size};">
            {txt}
        </p>
        """)
    else:
        s = (f"""
        <p style="font-size:{font_size}; color:{font_color}">
            {txt}
        </p>
        """)
    if center:
        s = re.sub(r'<p style="(.*)">', r'<p style="\1; text-align:center">', s)
    return s

# Syntax highlighting html
func_color = (111/255,66/255,205/255) # purplish
kwargs_color = (208/255,88/255,9/255) # reddish/orange
class_color = (215/255,58/255,73/255) # reddish
blue_color = (0/255,92/255,197/255) # blueish
class_sh = span('<i>class</i>', color=class_color)
def_sh = span('<i>def</i>', color=class_color)
if_sh = span('<i>if</i>', color=class_color)
elif_sh = span('<i>elif</i>', color=class_color)
kwargs_sh = span('**kwargs', color=kwargs_color)
Model_sh = span('Model', color=func_color)
segment_sh = span('segment', color=func_color)
predict_sh = span('predict', color=func_color)
CV_sh = span('CV', color=func_color)
init_sh = span('__init__', color=blue_color)
myModel_sh = span('MyModel', color=func_color)
return_sh = span('<i>return</i>', color=class_color)
equal_sh = span('=', color=class_color)
open_par_sh = span('(', color=blue_color)
close_par_sh = span(')', color=blue_color)
image_sh = span('image', color=kwargs_color)
from_sh = span('<i>from</i>', color=class_color)
import_sh = span('<i>import</i>', color=class_color)
is_not_sh = span('is not', color=class_color)
np_mean_sh = span('np.mean', color=class_color)
np_std_sh = span('np.std', color=class_color)