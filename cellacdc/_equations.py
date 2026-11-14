"""Shared input-token detection for pandas measurement equations."""

import keyword
import tokenize
from io import StringIO


def _equation_tokens(equation: str, offsets: list[int]) -> list:
    """Tokens of ``equation`` up to the first tokenizer error.

    Live editing can leave an incomplete expression (evaluation reports
    it). Python >= 3.12 also stops at backtick-quoted names with a number
    followed by letters or ``_`` at their start or after punctuation
    (```561_mean```, ```GFP-488_mean```; 3.10/3.11 split them): the
    content of such a name is masked with ``_`` (same offsets) and
    tokenizing goes on, so the names after it are found too (evaluation
    then reports the unparseable name). ``offsets`` are the character
    offsets of the lines of ``equation``.
    """
    masked = equation
    while True:
        tokens = []
        try:
            tokens.extend(tokenize.generate_tokens(StringIO(masked).readline))
            return tokens
        except (tokenize.TokenError, IndentationError):
            pass
        quotes = [token for token in tokens if token.string == '`']
        if len(quotes) % 2 == 0:
            return tokens
        row, col = quotes[-1].start
        start = offsets[row-1] + col
        end = masked.find('`', start+1)
        content = masked[start+1:end]
        if end < 0 or '\n' in content or not content.strip('_'):
            return tokens
        masked = masked[:start+1] + '_'*len(content) + masked[end:]


def equation_variable_spans(equation: str) -> list[tuple[str, int, int]]:
    """Find input tokens without matching string literals or function names."""
    offsets = [0]
    for line in equation.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    tokens = _equation_tokens(equation, offsets)
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
