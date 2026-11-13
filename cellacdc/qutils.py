from  typing import Literal
import re
import textwrap
import functools
from qtpy.QtCore import QObject, Signal

from qtpy.QtCore import (
    Qt, QTimer, QEventLoop
)
from qtpy.QtWidgets import QWidget, QAction, QToolButton


class QWhileLoop:
    def __init__(
            self, loop_callback, period=100, max_duration=None
        ):
        self._loop_callback = loop_callback
        self._period = period
        self._max_duration = max_duration

    def exec_(self):
        self.loop = QEventLoop()
        self.timer = QTimer()
        self.timer.timeout.connect(self._loop_callback)
        self.timer.start(self._period)
        if self._max_duration is not None:
            self.max_duration_timer = QTimer()
            self.max_duration_timer.timeout.connect(self.stop)
            self.max_duration_timer.start(self._max_duration)
        self.loop.exec_()
    
    def stop(self):
        self.timer.stop()
        if self._max_duration is not None:
            self.max_duration_timer.stop()
        self.loop.exit()

class QControlBlink(QObject):
    """Blink one control or a group in sync, restoring each original style.

    Starting a blink on a widget that is still blinking stops the running
    blink first, so that its original style (not the blink color) is
    restored at the end.
    """
    # Running blink of each widget {widget: QControlBlink}
    _active = {}

    def __init__(
            self,
            widget_to_blink: (
                QWidget | QAction
                | list[QWidget | QAction] | tuple[QWidget | QAction, ...]
            ),
            duration_ms=2000,
            qparent=None,
        ) -> None:
        self.duration_ms = duration_ms
        self.qparent = qparent
        self.blinkON = False
        targets = (
            widget_to_blink
            if isinstance(widget_to_blink, (list, tuple))
            else (widget_to_blink,)
        )
        self._widgets = []
        for target in targets:
            if isinstance(target, QAction):
                associated_objects = (
                    target.associatedObjects()
                    if hasattr(target, 'associatedObjects')
                    else target.associatedWidgets()
                )
                widgets = [
                    widget for widget in associated_objects
                    if isinstance(widget, QToolButton)
                ]
            elif isinstance(target, QWidget):
                widgets = [target]
            else:
                raise TypeError('Blink targets must be widgets or actions')
            for widget in widgets:
                if widget not in self._widgets:
                    self._widgets.append(widget)
        for widget in self._widgets:
            running = QControlBlink._active.get(widget)
            if running is None:
                continue
            try:
                running.stop()
            except RuntimeError:
                # Its widgets or timers were already deleted
                QControlBlink._active.pop(widget, None)
        self._stopped = False
        self._original_styles = [
            widget.styleSheet() for widget in self._widgets
        ]
        super().__init__(qparent)
    
    def start(self):
        for widget in self._widgets:
            QControlBlink._active[widget] = self
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerCallback)
        self.timer.start(100)

        self.stopTimer = QTimer(self)
        self.stopTimer.setSingleShot(True)
        self.stopTimer.timeout.connect(self.stop)
        self.stopTimer.start(self.duration_ms)
    
    def _setStyleSheet(self, style):
        for widget in self._widgets:
            widget.setStyleSheet(style)

    def timerCallback(self):
        if self.blinkON:
            self._setStyleSheet('background-color: orange')
        else:
            self._setStyleSheet('background-color: none')
        self.blinkON = not self.blinkON

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.timer.stop()
        self.stopTimer.stop()
        for widget, style in zip(self._widgets, self._original_styles):
            if QControlBlink._active.get(widget) is self:
                del QControlBlink._active[widget]
            widget.setStyleSheet(style)
        self.deleteLater()
        
def hide_and_delete_layout(layout):
    # Hide all widgets in the layout
    for i in reversed(range(layout.count())):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            widget.hide()
            layout.removeWidget(widget)
            widget.setParent(None)
    
    # Delete the layout
    layout.deleteLater()

def delete_widget(widget):
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()

def replace_certain_vals(getVal, replace_val, by_val):
    """
    Decorator: If the return value of getVal equals replace_val (type-cast to value's type),
    return by_val instead. Otherwise, return the original value.
    """
    @functools.wraps(getVal)
    def wrapper(*args, **kwargs):
        value = getVal(*args, **kwargs)
        try:
            target_val = type(value)(replace_val)
        except Exception:
            return value
        if value == target_val:
            return by_val
        return value
    return wrapper

def set_value_no_signals(widget, value):
    was_blocked = widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(was_blocked)

def set_exclusive_valueSetter(widget, valueSetter, value):
    was_blocked = widget.blockSignals(True)
    try:
        valueSetter(widget, value)
    except Exception as err:
        valueSetter(value)
    widget.blockSignals(was_blocked)

def hardDelete(item, setPosData=True):
    try:
        item.setParent(None)
    except (AttributeError, RuntimeError):
        pass
    if setPosData:
        try:
            item.posData = None
        except:
            pass
    try:
        item.deleteLater()
    except (AttributeError, RuntimeError):
        pass
    item = None
    
def insert_row(layout, insert_at, new_widget, col=0, dont_shift_other_cols=False):
# Shift all widgets down by one row from insert_at onwards
    for row in range(layout.rowCount() - 1, insert_at - 1, -1):
        for loc_col in range(layout.columnCount()):
            if loc_col != col and dont_shift_other_cols:
                continue
            item = layout.itemAtPosition(row, loc_col)
            if item is not None:
                layout.removeItem(item)
                layout.addItem(item, row + 1, loc_col)
    layout.addWidget(new_widget, insert_at, col)

def tryDisconnectSignal(
        widget, 
        signal, 
        error: Literal['ignore', 'raise'] ='ignore'
    ):
    try:
        getattr(widget, signal).disconnect()
    except Exception as err:
        if error == 'raise':
            raise err


TOOLTIP_MAX_LINE_LENGTH = 80
"""Maximum number of characters of a plain-text tool tip line.

Qt never wraps plain-text tool tips (a single line can span the screen),
only rich text (at about 80 characters), so fixed tool tip texts get explicit
``'\\n'`` at natural breaks, and texts with content not known in advance
(names, descriptions or equations entered by the user) go through
`wrap_tooltip`.
"""

# HTML elements that make Qt treat a text as rich text (`might_be_rich_text`,
# the elements of Qt's ``QTextHtmlParser``)
_RICH_TEXT_TAGS = frozenset((
    'a', 'address', 'b', 'big', 'blockquote', 'body', 'br', 'caption',
    'center', 'cite', 'code', 'dd', 'dfn', 'div', 'dl', 'dt', 'em', 'font',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'hr', 'html', 'i', 'img',
    'kbd', 'li', 'link', 'meta', 'nobr', 'ol', 'p', 'pre', 'qt', 's', 'samp',
    'script', 'small', 'span', 'strong', 'style', 'sub', 'sup', 'table',
    'tbody', 'td', 'tfoot', 'th', 'thead', 'title', 'tr', 'tt', 'u', 'ul',
    'var'
))


def _might_be_rich_text_heuristic(text):
    """Port of Qt's ``Qt::mightBeRichText`` (used when Qt does not expose
    it): leading whitespace is skipped, then the first ``<`` before the first
    line break must open a known tag (`_RICH_TEXT_TAGS`)."""
    n = len(text)
    start = n - len(text.lstrip())
    if text[start:start+5] == '<?xml':
        end = text.find('?>', start)
        start = n if end == -1 else end + 2
        start = n - len(text[start:].lstrip())
    if text[start:start+5].lower() == '<!doc':
        return True
    open_ = start
    while open_ < n and text[open_] not in '<\n':
        if text[open_] == '&' and text[open_+1:open_+4] == 'lt;':
            return True
        open_ += 1
    if open_ >= n or text[open_] != '<':
        return False
    close = text.find('>', open_)
    if close == -1:
        return False
    tag = ''
    for i in range(open_ + 1, close):
        char = text[i]
        if char.isalnum():
            tag += char
        elif tag and char.isspace():
            break
        elif tag and char == '/' and i + 1 == close:
            break
        elif not char.isspace() and (tag or char != '!'):
            return False
    return tag.lower() in _RICH_TEXT_TAGS


def might_be_rich_text(text):
    """Whether Qt displays `text` (e.g., a tool tip) as rich text.

    Uses ``Qt.mightBeRichText`` (in ``QtCore.Qt`` with PyQt5/PyQt6,
    ``QtGui.Qt`` with PySide), else the same heuristic
    (`_might_be_rich_text_heuristic`): after leading whitespace, a known HTML
    tag before the first line break. Text with tags after the first line
    break is plain text, with the tags shown as they are.
    """
    func = getattr(Qt, 'mightBeRichText', None)
    if func is None:
        from qtpy import QtGui
        func = getattr(getattr(QtGui, 'Qt', None), 'mightBeRichText', None)
    if func is not None:
        return bool(func(text))
    return _might_be_rich_text_heuristic(text)


_LIST_ITEM_PREFIX = re.compile(r'\s*(?:[-*•]\s+|\d+[.)]\s+)?')


def balanced_wrap(line, width=TOOLTIP_MAX_LINE_LENGTH, indent=''):
    """Wrap `line` at spaces into the fewest lines of at most `width`
    characters, with lengths as even as possible.

    Greedy wrapping fills every line up to `width` and leaves a short last
    line; here the same number of lines is made with the narrowest width
    that still needs no extra line, so the lines are about equally long.

    Parameters
    ----------
    line : str
        Text without line breaks. Its leading whitespace stays at the start
        of the first line.
    width : int, optional
        Maximum line length. Default is `TOOLTIP_MAX_LINE_LENGTH`.
    indent : str, optional
        Prefix of the continuation lines (e.g., to align a list item).

    Returns
    -------
    list of str
        The lines. Words longer than `width` and hyphenated words are not
        split.
    """
    # Keep the leading whitespace as the first line's prefix: textwrap would
    # drop it (no extra line) at widths where it does not fit with the first
    # word
    stripped = line.lstrip()
    lead = line[:len(line) - len(stripped)]
    def wrap(w):
        return textwrap.wrap(
            stripped, w, initial_indent=lead, subsequent_indent=indent,
            break_long_words=False, break_on_hyphens=False
        )
    lines = wrap(width)
    if len(lines) <= 1:
        return lines or [line]
    shortest = max(len(indent) + 1, -(-len(line) // len(lines)))
    for narrower in range(shortest, width):
        candidate = wrap(narrower)
        if len(candidate) == len(lines):
            return candidate
    return lines


def wrap_tooltip(text, width=TOOLTIP_MAX_LINE_LENGTH):
    """Break the lines of a plain-text tool tip that are longer than `width`
    characters at spaces (Qt never wraps plain-text tool tips), into lines
    of about equal length (`balanced_wrap`).

    Existing line breaks, blank lines and indentation are kept; the
    continuation lines of a list item (``- ``, ``* ``, ``1. ``) are aligned
    with its text. Words longer than `width` (e.g., a column name) and
    hyphenated words are not split. Rich text (`might_be_rich_text`) is
    returned unchanged, because Qt word-wraps it.

    Parameters
    ----------
    text : str
        Tool tip text.
    width : int, optional
        Maximum line length. Default is `TOOLTIP_MAX_LINE_LENGTH`.

    Returns
    -------
    str
        The text with line breaks.
    """
    if not text or might_be_rich_text(text):
        return text
    lines = []
    for line in text.split('\n'):
        if len(line) <= width:
            lines.append(line)
            continue
        indent = ' ' * len(_LIST_ITEM_PREFIX.match(line).group())
        lines.extend(balanced_wrap(line, width, indent=indent))
    return '\n'.join(lines)
