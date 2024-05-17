from functools import wraps
import html
import re
import sys
import textwrap

from . import GUI_INSTALLED, myutils

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

ADMONITION_TYPES = (
    'topic', 
    'admonition', 
    'attention', 
    'caution', 
    'danger', 
    'error', 
    'hint', 
    'important', 
    'note', 
    'seealso', 
    'tip', 
    'todo', 
    'warning', 
    'versionadded', 
    'versionchanged', 
    'deprecated'
)

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

def span(text, color='r', font_size=None, bold=False):
    span_text = f'<span style="">{text}</span>'
    if color is not None:
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
        span_text = f'<span style="color: {c}">{text}</span>'
    if font_size is not None:
        span_text = span_text.replace('">', f'; font-size: {font_size};">')
    if bold:
        span_text = span_text.replace('">', f'; font-weight:bold;">')
    return span_text

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

def rst_to_html(rst_text, parse_urls=False, keep_spacing=False):
    if parse_urls:
        rst_text = rst_urls_to_html(rst_text)
    valid_chars = r'[,A-Za-z0-9μ\-\.=_ \<\>\(\)\\\&;]'
    html_text = re.sub(rf'\`\`([^\`]*)\`\`', r'<code>\1</code>', rst_text)
    html_text = re.sub(rf'\`([^\`]*)\`', r'<code>\1</code>', html_text)
    html_text = html_text.replace('\n', '<br>')
    html_text = html_text.replace('<', '&lt;').replace('>', '&gt;')
    if keep_spacing:
        html_text = re.sub(
            r'(\s\s+)', lambda m: '&nbsp;'*len(m.group(0)), html_text
        )
    return html_text

def rst_docstring_filter_args(rst_doc, args_to_keep):
    start_idx = rst_doc.find('Parameters')
    before_params_text = rst_doc[:start_idx]
    start_params_idx = before_params_text.rfind('\n') + 1
    
    params_text = rst_doc[start_params_idx:]    
    numls = len(params_text) - len(params_text.lstrip())
    ul = ' '*numls + '-'*len('Parameters')
    section = ' '*numls + 'Parameters'
    section_header = f'{section}\n{ul}\n'
    
    found_end = re.findall(r'\n *\n', params_text)
    if not found_end:
        stop_idx = None
    else:
        stop_idx = params_text.find(found_end[0])
    
    params_text = params_text[:stop_idx]
    filtered_params_text = params_text
    found_args = re.findall(r'([A-Za-z0-9_]+) \: (.*)', params_text)
    for a, (arg_name, arg_dtype) in enumerate(found_args):
        if arg_name in args_to_keep:
            continue
        
        arg_doc = f' {arg_name} : {arg_dtype}'
        start_idx = filtered_params_text.find(arg_doc) + 1
        
        if a+1 == len(found_args):
            stop_idx = None
        else:
            next_arg, next_arg_type = found_args[a+1]
            next_arg_doc = f' {next_arg} : {next_arg_type}'
            stop_idx = filtered_params_text.find(next_arg_doc)
        
        text_to_remove = filtered_params_text[start_idx:stop_idx]
        filtered_params_text = filtered_params_text.replace(text_to_remove, '')
    
    filtered_params_text = filtered_params_text.rstrip().rstrip('\n')
    filtered_doc = rst_doc.replace(params_text, filtered_params_text)
    return filtered_doc

def rst_docstring_to_html(rst_doc: str, args_subset=None):
    html_text = rst_doc
    
    if args_subset is not None:
        html_text = rst_docstring_filter_args(html_text, args_subset)
    
    # Replace args with indented `bold : italic`
    found_args = re.findall(r'([A-Za-z0-9_]+) \: (.*)', html_text)
    for a, (arg_name, arg_dtype) in enumerate(found_args):
        arg_doc = f' {arg_name} : {arg_dtype}'        
        html_text = html_text.replace(
            arg_doc, 
            f'<br>&nbsp;&nbsp;<b>{arg_name} : <i>{arg_dtype}</i></b>',
        )
    
    # Indent description of arg more
    admon_sections = []
    found_sections = re.findall(r'([A-Za-z ]+)\n *[\-]+\n', rst_doc)
    for s, section in enumerate(found_sections):
        section_lstrip = section.lstrip()
        section_admon = section_lstrip.replace(' ', '').lower()
        if section_admon in ADMONITION_TYPES:
            admon_sections.append(section)
            continue
        
        numls = len(section) - len(section_lstrip)
        ul = ' '*numls + '-'*len(section_lstrip)
        section_header = f'{section}\n{ul}\n'
        start_idx = html_text.find(section_header) + len(section_header)
        if s+1 == len(found_sections):
            stop_idx = None
        else:
            next_section = found_sections[s+1]
            stop_idx = html_text.find(next_section)
        
        section_text = html_text[start_idx:stop_idx]
        section_indented = re.sub(
            r'(\n\s\s+)', '<br>&nbsp;&nbsp;&nbsp;&nbsp;', section_text
        )
        
        html_text = list(html_text)
        html_text[start_idx:stop_idx] = section_indented
        html_text = ''.join(html_text)
    
    # Replace section header with 16px bold html
    for section in found_sections:
        if section in admon_sections:
            continue
        
        section_lstrip = section.lstrip()       
        numls = len(section) - len(section_lstrip)
        ul = ' '*numls + '-'*len(section_lstrip)
        html_text = html_text.replace(
            f'{section}\n{ul}', 
            span(section.strip(), font_size='16px', color=None, bold=True)
        )
    
    # Replace admonition sections with html table
    for admon_section in admon_sections:
        section_lstrip = admon_section.lstrip()       
        numls = len(admon_section) - len(section_lstrip)
        ul = ' '*numls + '-'*len(section_lstrip)
        section_header = f'{admon_section}\n{ul}\n'
        
        start_idx = html_text.find(section_header) + len(section_header)
        section_text = html_text[start_idx:]
        found_end = re.findall(r'\n *\n', section_text)
        if not found_end:
            stop_idx = None
        else:
            stop_idx = section_text.find(found_end[0])
        
        section_text = section_text[:stop_idx]
        html_admon = to_admonition(section_text, admonition_type=section_lstrip)
        html_text = html_text.replace(section_text, html_admon)
        html_text = html_text.replace(section_header, '')
    
    # Replace last charachaters to html
    html_text = rst_urls_to_html(html_text)
    html_text = html_text.replace('\n', '<br>')
    html_text = re.sub(rf'\`\`([^\`]*)\`\`', r'<code>\1</code>', html_text)
    html_text = re.sub(rf'\`([^\`]*)\`', r'<code>\1</code>', html_text)
    
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