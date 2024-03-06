from functools import wraps
import html
import re
import sys
import textwrap

from . import GUI_INSTALLED

from ._palettes import (
    _get_highligth_header_background_rgba, _get_highligth_text_background_rgba
)
from .colors import rgb_uint_to_html_hex

if GUI_INSTALLED:
    from matplotlib.colors import to_hex

is_mac = sys.platform == 'darwin'

RST_NOTE_DIR_RGBA = _get_highligth_header_background_rgba()
RST_NOTE_DIR_HEX_COLOR = rgb_uint_to_html_hex(RST_NOTE_DIR_RGBA[:3])

RST_NOTE_TXT_RGBA = _get_highligth_text_background_rgba()
RST_NOTE_TXT_HEX_COLOR = rgb_uint_to_html_hex(RST_NOTE_TXT_RGBA[:3])

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

def to_plain_text(html_text):
    html_text = re.sub(r' +', ' ', html_text)
    html_text = html_text.replace('\n ', '\n')
    html_text = html_text.strip('\n')
    html_text = html_text.replace('<code>', '`')
    html_text = html_text.replace('</code>', '`')
    html_text = html_text.replace('<br>', '\n')
    html_text = html_text.replace('<li>', '\n  * ')
    html_text = re.sub(r'</\w+>', '', html_text)
    html_text = re.sub(r'<.+>', '', html_text)
    html_text = html_text.strip('\n')
    return html_text

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

def rst_urls_to_html(rst_text):
    links = re.findall(r'`(.*) ?<(.*)>`_', rst_text)
    html_text = rst_text
    for text, link in links:
        if not text:
            text = link
        repl = href_tag(text.rstrip(), link)
        pattern = fr'`{text} ?<{link}>`_'
        html_text = re.sub(pattern, repl, html_text)
    return html_text

def to_admonition(text, admonition_type='note'):
    if text.find('<br>') == -1:
        wrapped_list = textwrap.wrap(text, width=130)
        text = '<br>'.join(wrapped_list)
    title = admonition_type.capitalize()
    title_row = tag(
        f'<td><b>! {title}</b></td>', 
        tag_info=f'tr bgcolor="{RST_NOTE_DIR_HEX_COLOR}"'
    )
    text_row = tag(
        f'<td>{text}</td>', 
        tag_info=f'tr bgcolor="{RST_NOTE_TXT_HEX_COLOR}"'
    )
    admonition_html = (
        '<table cellspacing=0 cellpadding=5 width=100%>'
        f'{title_row}{text_row}'
        '</table><br>'
    )
    return admonition_html

def to_note(note_text):
    note_html = to_admonition(note_text, admonition_type='note')
    return note_html

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