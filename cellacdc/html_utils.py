from functools import wraps
import re
from matplotlib.colors import to_hex

from . import is_mac

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
    c = to_hex(color)
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
