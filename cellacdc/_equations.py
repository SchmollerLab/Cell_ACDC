"""Shared input-token detection for pandas measurement equations."""

import keyword
import tokenize
from io import StringIO


def equation_variable_spans(equation: str) -> list[tuple[str, int, int]]:
    """Find input tokens without matching string literals or function names."""
    offsets = [0]
    for line in equation.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    tokens = []
    try:
        tokens.extend(tokenize.generate_tokens(StringIO(equation).readline))
    except (tokenize.TokenError, IndentationError):
        # Live editing can leave an incomplete expression; evaluation reports it.
        pass
    spans = []
    quoted_start = None
    for i, token in enumerate(tokens):
        start = offsets[token.start[0]-1] + token.start[1]
        end = offsets[token.end[0]-1] + token.end[1]
        if token.string == '`':
            if quoted_start is None:
                quoted_start = start
            else:
                spans.append((equation[quoted_start+1:start], quoted_start, end))
                quoted_start = None
        elif quoted_start is None and token.type == tokenize.NAME:
            previous = tokens[i-1].string if i else ''
            following = tokens[i+1].string if i+1 < len(tokens) else ''
            if (
                not keyword.iskeyword(token.string)
                and previous != '.' and following != '('
            ):
                spans.append((token.string, start, end))
    return spans
