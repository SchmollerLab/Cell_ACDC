"""Byte-exact source handling.

SourceText keeps a file's encoding, BOM and line endings: every line keeps
its own end-of-line characters (the working tree has CRLF with
core.autocrlf, the object store LF), so text that is moved stays
byte-identical. Generated lines use the file's dominant EOL.

`ast` column offsets are UTF-8 byte offsets; `char_col` converts them.

FileEdit collects edits in the coordinates of the original text (1-based
lines, character columns) and renders them at once:

- splice(line, start, end, text): replace part of one line;
- replace_lines(first, last, lines): replace whole lines;
- delete_lines(first, last);
- insert_before(line, lines): insert whole lines (line = n + 1 appends).
"""
import codecs
import io
import re
import tokenize

from . import RefactorError

_LINE_RE = re.compile(r'[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+\Z')


def split_lines(text):
    """Split keeping line ends; only \\r\\n, \\n and \\r end a line (as for
    the Python tokenizer, unlike str.splitlines)."""
    return _LINE_RE.findall(text)


def line_end(line):
    if line.endswith('\r\n'):
        return '\r\n'
    if line.endswith('\n'):
        return '\n'
    if line.endswith('\r'):
        return '\r'
    return ''


def strip_eol(line):
    return line[:len(line) - len(line_end(line))]


def char_col(line, byte_col):
    """Character column of the UTF-8 byte column `byte_col` in `line`."""
    if line.isascii():
        return byte_col
    return len(line.encode('utf-8')[:byte_col].decode('utf-8', 'replace'))


def indent_of(line):
    s = strip_eol(line)
    return len(s) - len(s.lstrip(' \t'))


def is_blank(line):
    return not strip_eol(line).strip()


def is_comment(line):
    return strip_eol(line).lstrip().startswith('#')


class SourceText:
    def __init__(self, text, encoding='utf-8', bom=False, eol=None):
        self.text = text
        self.encoding = encoding
        self.bom = bom
        self.lines = split_lines(text)
        self.eol = eol or self._dominant_eol()

    def _dominant_eol(self):
        counts = {}
        for line in self.lines:
            e = line_end(line)
            if e:
                counts[e] = counts.get(e, 0) + 1
        if not counts:
            return '\n'
        return max(sorted(counts), key=lambda e: counts[e])

    @classmethod
    def from_bytes(cls, data):
        bom = data.startswith(codecs.BOM_UTF8)
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        except SyntaxError as err:
            raise RefactorError(f'Cannot detect the encoding: {err}')
        if encoding == 'utf-8-sig':
            encoding = 'utf-8'
        raw = data[len(codecs.BOM_UTF8):] if bom else data
        return cls(raw.decode(encoding), encoding=encoding, bom=bom)

    @classmethod
    def read(cls, path):
        with open(path, 'rb') as file:
            return cls.from_bytes(file.read())

    def to_bytes(self):
        data = self.text.encode(self.encoding)
        if self.bom:
            data = codecs.BOM_UTF8 + data
        return data

    def write(self, path):
        with open(path, 'wb') as file:
            file.write(self.to_bytes())

    def with_text(self, text):
        return SourceText(
            text, encoding=self.encoding, bom=self.bom, eol=self.eol
        )

    def line(self, lineno):
        return self.lines[lineno - 1]

    def segment(self, first, last):
        """Text of lines first..last (1-based, inclusive)."""
        return ''.join(self.lines[first - 1:last])


def normalize_eol(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')


def ensure_eol(lines, eol):
    """Make sure the last line ends with an EOL."""
    if lines and not line_end(lines[-1]):
        lines[-1] = lines[-1] + eol
    return lines


def text_to_lines(text, eol):
    """Generated text (with \\n) to lines with the file's EOL."""
    if not text:
        return []
    return [strip_eol(line) + eol for line in split_lines(text)]


class FileEdit:
    """Edits of one file in original coordinates, rendered at once."""

    def __init__(self, src, path=None):
        self.src = src
        self.path = path
        self.eol = src.eol
        self._splices = {}          # line -> [(start, end, text)]
        self._replaced = {}         # first line -> (last, [lines])
        self._deleted = set()
        self._inserts = {}          # line -> [lines]

    @property
    def n_lines(self):
        return len(self.src.lines)

    def _check_free(self, first, last):
        for line in range(first, last + 1):
            if line in self._deleted:
                raise RefactorError(
                    f'{self.path}: overlapping edits at line {line}'
                )
            for f, (l, _) in self._replaced.items():
                if f <= line <= l:
                    raise RefactorError(
                        f'{self.path}: overlapping edits at line {line}'
                    )

    def splice(self, lineno, start, end, text):
        self._splices.setdefault(lineno, []).append((start, end, text))

    def replace_lines(self, first, last, lines):
        self._check_free(first, last)
        self._replaced[first] = (last, list(lines))

    def delete_lines(self, first, last):
        self._check_free(first, last)
        self._deleted.update(range(first, last + 1))

    def insert_before(self, lineno, lines):
        self._inserts.setdefault(lineno, []).extend(lines)

    def is_deleted(self, lineno):
        return lineno in self._deleted

    def _render_line(self, lineno):
        line = self.src.lines[lineno - 1]
        splices = self._splices.get(lineno)
        if not splices:
            return line
        eol = line_end(line)
        body = line[:len(line) - len(eol)]
        last_start = None
        for start, end, text in sorted(splices, reverse=True):
            if last_start is not None and end > last_start:
                raise RefactorError(
                    f'{self.path}: overlapping splices at line {lineno}'
                )
            body = body[:start] + text + body[end:]
            last_start = start
        return body + eol

    def render_range(self, first, last, include_deleted=True):
        """Lines first..last with splices and line replacements applied
        (insertions and deletions are ignored unless include_deleted is
        False, in which case deleted lines are skipped)."""
        out = []
        lineno = first
        while lineno <= last:
            if lineno in self._replaced:
                end, lines = self._replaced[lineno]
                out.extend(lines)
                lineno = end + 1
                continue
            if not include_deleted and lineno in self._deleted:
                lineno += 1
                continue
            out.append(self._render_line(lineno))
            lineno += 1
        return out

    def render(self):
        out = []
        n = self.n_lines
        lineno = 1
        while lineno <= n:
            out.extend(self._inserts.get(lineno, []))
            if lineno in self._replaced:
                end, lines = self._replaced[lineno]
                if lineno not in self._deleted:
                    out.extend(lines)
                lineno = end + 1
                continue
            if lineno not in self._deleted:
                out.append(self._render_line(lineno))
            lineno += 1
        tail = self._inserts.get(n + 1, [])
        if tail:
            if out:
                ensure_eol(out, self.eol)
            out.extend(tail)
        return ''.join(out)

    def result(self):
        return self.src.with_text(self.render())
