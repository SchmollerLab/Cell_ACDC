"""Text helpers shared by move.py and shims.py."""
import re
import textwrap

_LIST_ITEM = re.compile(r'^(\s*)([-*]|\d+\.)\s+')


def _wrap_paragraph(lines, width):
    """Wrap a paragraph; list items ('- x', '* x', '1. x') are wrapped one
    by one with a hanging indent, indented code-like lines are kept."""
    out = []
    current = None      # (prefix, hang, [words])

    def flush():
        if current is not None:
            prefix, hang, words = current
            out.append(textwrap.fill(
                ' '.join(words), width=width, initial_indent=prefix,
                subsequent_indent=hang, break_on_hyphens=False,
                break_long_words=False))

    for line in lines:
        match = _LIST_ITEM.match(line)
        if match:
            flush()
            prefix = match.group(0)
            current = (prefix, ' ' * len(prefix),
                       [line[len(prefix):].strip()])
        elif line.startswith('    ') or line.lstrip().startswith('>>>'):
            flush()
            current = None
            out.append(line.rstrip())
        elif current is not None:
            current[2].append(line.strip())
        else:
            current = ('', '', [line.strip()])
    flush()
    return '\n'.join(out)


def format_docstring(doc, indent=''):
    """A triple-quoted docstring with paragraphs wrapped at 72 columns."""
    doc = (doc or '').strip()
    paragraphs = re.split(r'\n\s*\n', doc)
    width = max(40, 72 - len(indent))
    text = '\n\n'.join(_wrap_paragraph(p.split('\n'), width)
                       for p in paragraphs)
    quote = '"""' if '"""' not in text else "'''"
    prefix = 'r' if '\\' in text else ''
    if '\n' in text or len(indent) + len(text) + 6 > 79:
        body = text.replace('\n', '\n' + indent)
        result = f'{indent}{prefix}{quote}{body}\n{indent}{quote}'
    else:
        result = f'{indent}{prefix}{quote}{text}{quote}'
    # blank lines inside the docstring must not carry the indentation
    return '\n'.join(line.rstrip() for line in result.split('\n'))
