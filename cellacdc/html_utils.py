from . import is_mac

def css_head(txt):
    if is_mac:
        txt = txt.replace(',', ',&nbsp;')
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

def paragraph(txt, font_size='13px', font_color=None):
    if is_mac:
        txt = txt.replace(',', ',&nbsp;')
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
    return s
