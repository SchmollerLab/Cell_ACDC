def css_head(txt):
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
    s = (f"""
    <body>
        {txt}
    </body>
    </html>
    """)
    return s

def paragraph(txt, font_size='13px'):
    s = (f"""
    <p style="font-size:{font_size};">
        {txt}
    </p>
    """)
    return s
