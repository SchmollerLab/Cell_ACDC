"""Tool tips of the measurements viewer must fit on lower resolution screens.

Qt never wraps plain-text tool tips (each line is as wide as its text), only
rich text (at about 80 characters). Fixed texts therefore get explicit
``'\\n'`` and texts with content not known in advance (names, descriptions,
equations) go through `qutils.wrap_tooltip`, so that no plain-text line is
longer than `qutils.TOOLTIP_MAX_LINE_LENGTH`.

The viewer window and its dialogs are built with a small synthetic table and
dynamic content (time point identifiers, custom annotations, conditional
formatting and phase rules of the premade FUCCI configuration, plus a long
description and a long condition), then every tool tip they can show is
collected (widgets, actions, item models, plot items, hover tips of the
plotted points) and checked. Dialogs are opened with the viewer's own
methods with ``exec_`` replaced, so they never block.
"""
import html
import json
import os
import re
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtWidgets import (
    QAbstractItemView, QAbstractButton, QAction, QApplication, QComboBox,
    QGraphicsItem, QGroupBox, QLabel, QToolButton, QWidget, QWidgetAction
)

from cellacdc import apps, apps_measurements_viewer, qutils
from cellacdc.measurements_viewer import (
    MeasurementsViewerWin, PREMADE_CONFIGURATIONS_PATH
)
from cellacdc.timepoint_predictors import TimepointIdentifier

MAX = qutils.TOOLTIP_MAX_LINE_LENGTH

LONG_DESCRIPTION = (
    'Cells that stop growing during the experiment and never bud again, '
    'for example because they ran out of nutrients or were damaged by the '
    'laser, and must be excluded from every downstream statistic.'
)
LONG_CONDITION = (
    '(area > 12) & (intensity > 120) & (area < 100000) '
    '& (intensity < 100000) & (area != 12345)'
)
LONG_EQUATION = (
    '(area > 1.5 * MEAN(area)) & (intensity > MIN(intensity) + 0.1 * '
    '(MAX(intensity) - MIN(intensity))) & (frame_i > FIRST(frame_i))'
)
GFP, MCHERRY = 'GFP', 'mCherry'


# --- Checks -----------------------------------------------------------------
# Tags that Qt would show literally in a plain-text tool tip (the name right
# after "<", so that conditions such as "x < a & y > 1" do not match)
_PLAIN_TEXT_TAG = re.compile(
    r'</?(?:a|b|i|u|s|br|p|code|li|ul|ol|pre|nobr|span|div|font|'
    r'table|tr|td|th|em|strong|tt|sub|sup|h[1-6])(?:\s[^<>]*)?\s*/?\s*>',
    re.IGNORECASE
)


def visible_rich_text_lines(text):
    """Lines of a rich text as displayed, with <br>, <p>, <li> (and raw new
    lines inside <pre>) as line breaks and the tags removed."""
    parts = re.split(r'(<pre\b.*?</pre\s*>)', text, flags=re.I | re.S)
    text = ''.join(
        part if re.match(r'<pre\b', part, re.I) else part.replace('\n', ' ')
        for part in parts
    )
    text = re.sub(
        r'<br\s*/?>|</?p\b[^<>]*>|<li\b[^<>]*>|</?div\b[^<>]*>|'
        r'</?tr\b[^<>]*>|</?h[1-6]\b[^<>]*>|</?pre\b[^<>]*>',
        '\n', text, flags=re.I
    )
    text = html.unescape(re.sub(r'<[^<>]*>', '', text))
    return [line.strip() for line in text.split('\n')]


def tooltip_problem(text):
    """Return ``(max line length, reason)`` if the tool tip `text` would be
    too wide or shows HTML tags literally, else None."""
    if qutils.might_be_rich_text(text):
        lower = text.lower()
        if not any(key in lower for key in ('<pre', '<nobr', 'white-space')):
            # Qt word-wraps rich text
            return None
        longest = max(map(len, visible_rich_text_lines(text)))
        if longest > MAX:
            return longest, 'unwrappable rich-text line'
        return None
    longest = max(map(len, text.split('\n')))
    reasons = []
    if longest > MAX:
        reasons.append('plain-text line too long')
    if _PLAIN_TEXT_TAG.search(text):
        reasons.append(
            'HTML tags in plain text (no tag before the first line break, '
            'so Qt shows them literally)'
        )
    if not reasons:
        return None
    return longest, ', '.join(reasons)


def check(tooltips):
    """Offender lines of the collected `{text: owner}` tool tips."""
    offenders = []
    for text, owner in tooltips.items():
        problem = tooltip_problem(text)
        if problem is None:
            continue
        longest, reason = problem
        offenders.append(
            f'- {owner}\n    {reason}, longest line {longest} chars: '
            f'{text[:120]!r}'
        )
    return offenders


def assert_no_offenders(tooltips):
    offenders = check(tooltips)
    assert not offenders, (
        f'{len(offenders)} tool tip(s) with lines longer than {MAX} '
        'characters or with HTML tags shown literally (add "\\n" to fixed '
        'texts, wrap dynamic ones with qutils.wrap_tooltip):\n'
        + '\n'.join(offenders)
    )


# --- Collection -------------------------------------------------------------
def _call(obj, method):
    try:
        value = getattr(obj, method)()
    except Exception:
        return ''
    return value if isinstance(value, str) else ''


def describe(obj):
    """Class name, object name and text/title of `obj` (for messages)."""
    label = type(obj).__name__
    name = _call(obj, 'objectName')
    if name:
        label += f' {name!r}'
    if isinstance(obj, (QAbstractButton, QAction, QLabel)):
        text = _call(obj, 'text')
    elif isinstance(obj, QGroupBox):
        text = _call(obj, 'title')
    elif isinstance(obj, QWidget) and obj.isWindow():
        text = _call(obj, 'windowTitle')
    else:
        text = ''
    text = ' '.join(text.split())
    if text:
        label += f' "{text[:50]}"'
    return label


def stripped_action_text(text):
    """Default tool tip of a QAction (``qt_strippedText``: text without
    "...", "\u2026" and mnemonic ampersands)."""
    text = text.replace('...', '').replace('\u2026', '')
    text = re.sub(r'&(.)', r'\1', text)
    return text.strip()


def column_count(model, parent):
    try:
        return model.columnCount(parent)
    except (TypeError, NotImplementedError, AttributeError):
        # Private in list models (e.g. QStringListModel): one column
        return 1


def model_tooltips(model, prefix, add, parent=QModelIndex(), depth=0):
    """Call ``add(tip, owner)`` for the ``ToolTipRole`` data of the header
    and of every item (all columns, recursively) of `model`."""
    if depth == 0:
        for section in range(column_count(model, QModelIndex())):
            tip = model.headerData(section, Qt.Horizontal, Qt.ToolTipRole)
            if isinstance(tip, str):
                add(tip, f'{prefix} header {section}')
    for row in range(model.rowCount(parent)):
        for col in range(column_count(model, parent)):
            index = model.index(row, col, parent)
            tip = index.data(Qt.ToolTipRole)
            if isinstance(tip, str):
                add(tip, f'{prefix} item {index.data()!r}')
        child = model.index(row, 0, parent)
        try:
            hasChildren = model.hasChildren(child)
        except (TypeError, NotImplementedError, AttributeError):
            # Private in list models (e.g. of QListWidget): flat
            hasChildren = False
        if hasChildren:
            model_tooltips(model, prefix, add, child, depth + 1)


def scene_tooltips(view, prefix, add):
    """Tool tips of the graphics items of the pyqtgraph `view` (e.g. lines,
    ViewBox) and the hover tips of the points of hoverable scatter items."""
    scene = view.scene()
    if scene is None:
        return
    for item in scene.items():
        tip = QGraphicsItem.toolTip(item)
        if tip:
            add(tip, f'{prefix} > {type(item).__name__}')
        if not isinstance(item, pg.ScatterPlotItem):
            continue
        tipFunc = item.opts.get('tip')
        if not item.opts.get('hoverable') or not callable(tipFunc):
            continue
        # Hover tip of each point (as `ScatterPlotItem.hoverEvent` builds it)
        for point in item.points():
            text = tipFunc(
                x=point.pos().x(), y=point.pos().y(), data=point.data()
            )
            add(text, f'{prefix} > hover tip of a point')


def collect_tooltips(window, tooltips=None):
    """Add the tool tips that `window` (and its children) can show to
    `tooltips` ({text: owner}) and return it."""
    if tooltips is None:
        tooltips = {}
    windowName = describe(window)

    def add(text, owner):
        if text and text not in tooltips:
            tooltips[text] = f'{windowName}: {owner}'

    visitedModels = set()

    def addModel(model, owner):
        if model is None or id(model) in visitedModels:
            return
        visitedModels.add(id(model))
        model_tooltips(model, owner, add)

    widgets, actions = all_widgets_and_actions(window)
    for widget in widgets:
        tip = QWidget.toolTip(widget)
        if isinstance(widget, QToolButton) and widget.defaultAction():
            # Set from the action (collected below) unless changed
            if tip == QAction.toolTip(widget.defaultAction()):
                tip = ''
        add(tip, describe(widget))
        if isinstance(widget, QAbstractItemView):
            addModel(widget.model(), describe(widget))
        elif isinstance(widget, QComboBox):
            addModel(widget.model(), describe(widget))
        if isinstance(widget, pg.GraphicsView):
            scene_tooltips(widget, describe(widget), add)
    for action in actions:
        tip = QAction.toolTip(action)
        # Without a tool tip of its own, an action shows its text
        if tip in (action.text(), stripped_action_text(action.text())):
            continue
        add(tip, describe(action))
    return tooltips


def all_widgets_and_actions(window):
    """Widgets and actions of `window`: its children, plus the actions added
    to them without being their children (e.g. ``QAction(text)`` added to a
    menu), the menus of these actions and the widgets of widget actions."""
    widgets, actions = {}, {}
    stack = [window]
    while stack:
        root = stack.pop()
        for widget in [root, *root.findChildren(QWidget)]:
            if id(widget) in widgets:
                continue
            widgets[id(widget)] = widget
            if isinstance(widget, QToolButton) and widget.menu() is not None:
                stack.append(widget.menu())
            for action in widget.actions():
                if id(action) in actions:
                    continue
                actions[id(action)] = action
                if action.menu() is not None:
                    stack.append(action.menu())
                if isinstance(action, QWidgetAction):
                    if action.defaultWidget() is not None:
                        stack.append(action.defaultWidget())
        for action in root.findChildren(QAction):
            if id(action) not in actions:
                actions[id(action)] = action
                if action.menu() is not None:
                    stack.append(action.menu())
    return list(widgets.values()), list(actions.values())


# --- wrap_tooltip / might_be_rich_text --------------------------------------
LONG_LINE = ' '.join(f'word{i}' for i in range(40))


def lines_fit(text, width=MAX):
    return all(len(line) <= width for line in text.split('\n'))


def test_wrap_tooltip_breaks_long_plain_lines_at_spaces():
    wrapped = qutils.wrap_tooltip(LONG_LINE)
    assert '\n' in wrapped and lines_fit(wrapped)
    assert wrapped.split() == LONG_LINE.split()
    assert wrapped.replace('\n', ' ') == LONG_LINE
    assert qutils.wrap_tooltip('Short tip') == 'Short tip'
    assert qutils.wrap_tooltip('') == ''


def test_wrap_tooltip_balances_line_lengths():
    # Greedy wrapping at 80 gives 80, 80 and a short last line; the
    # balanced lines are about equally long, with no extra line
    text = ' '.join(['abcdefghi'] * 18)
    greedy = textwrap.wrap(text, MAX)
    lines = qutils.wrap_tooltip(text).split('\n')
    assert len(lines) == len(greedy) == 3
    assert len(greedy[-1]) < 20
    lengths = [len(line) for line in lines]
    assert max(lengths) - min(lengths) <= 10
    assert ' '.join(lines) == text
    assert qutils.balanced_wrap('one two', 80) == ['one two']


def test_wrap_tooltip_keeps_line_breaks_and_blank_lines():
    text = f'Title\n\n{LONG_LINE}\n\nShortcut: "Q"'
    wrapped = qutils.wrap_tooltip(text)
    assert wrapped.startswith('Title\n\n')
    assert wrapped.endswith('\n\nShortcut: "Q"')
    assert lines_fit(wrapped)
    assert qutils.wrap_tooltip('A\n\nB\n') == 'A\n\nB\n'


@pytest.mark.parametrize('prefix', ['- ', '* ', '1. ', '    '])
def test_wrap_tooltip_aligns_list_item_continuation(prefix):
    wrapped = qutils.wrap_tooltip(f'Items:\n{prefix}{LONG_LINE}')
    first, *continuation = wrapped.split('\n')[1:]
    assert first.startswith(prefix) and continuation
    indent = ' ' * len(prefix)
    for line in continuation:
        assert line.startswith(indent) and not line[len(indent)].isspace()
    assert lines_fit(wrapped)


def test_wrap_tooltip_keeps_indent_of_a_long_first_word():
    # Narrower candidate widths must not drop the first line's indentation
    # (textwrap drops leading whitespace that does not fit with the word)
    wrapped = qutils.wrap_tooltip(
        'Details:\n    ' + 'x'*50 + ' ' + 'y'*50
    ).split('\n')
    assert wrapped[1] == '    ' + 'x'*50
    assert wrapped[2] == '    ' + 'y'*50
    assert qutils.wrap_tooltip('    ' + 'y'*100) == '    ' + 'y'*100


def test_wrap_tooltip_never_splits_long_or_hyphenated_words():
    word = 'measurement_' * 10
    assert qutils.wrap_tooltip(word) == word
    wrapped = qutils.wrap_tooltip(f'Column {word} is used')
    assert word in wrapped.split('\n')
    hyphenated = 'mKO2-hCdt1-mAG1-hGeminin'
    wrapped = qutils.wrap_tooltip(f'{LONG_LINE} {hyphenated} {LONG_LINE}')
    assert any(hyphenated in line for line in wrapped.split('\n'))
    assert lines_fit(wrapped)


def test_wrap_tooltip_returns_rich_text_unchanged():
    text = f'<b>Title</b> {LONG_LINE}'
    assert qutils.might_be_rich_text(text)
    assert qutils.wrap_tooltip(text) == text
    text = f'<p>{LONG_LINE}</p>'
    assert qutils.wrap_tooltip(text) == text


def test_might_be_rich_text_needs_a_tag_on_the_first_line():
    assert qutils.might_be_rich_text('<b>Bold</b> text')
    assert qutils.might_be_rich_text('Text<br>more')
    assert not qutils.might_be_rich_text('Plain text')
    assert not qutils.might_be_rich_text('area < 500 & intensity > 2')
    # Tags only after the first line break: plain text, tags shown as is
    text = f'Title\n<b>{LONG_LINE}</b>'
    assert not qutils.might_be_rich_text(text)
    wrapped = qutils.wrap_tooltip(text)
    assert wrapped != text and lines_fit(wrapped)


@pytest.mark.parametrize('text', [
    '', 'Plain text', ' <b>x</b>', ' <b>x</b>', '<?xml version="1.0"?><b>x',
    '<!DOCTYPE html>', 'a &lt; b', 'Text<br>more', '<br/>', '<B>x</B>',
    '< b>x', '<!b>x', '<1b>x', '<del>x', '<font color=red>x', '<qt>x',
    '<h7>x', 'a<b', 'x\n<b>y</b>', 'phase_<name>', 'area < 500 & i > 2',
])
def test_might_be_rich_text_heuristic_matches_qt(text):
    # The heuristic is the fallback for bindings without Qt.mightBeRichText
    func = getattr(Qt, 'mightBeRichText', None)
    if func is None:
        pytest.skip('Qt.mightBeRichText not exposed by this binding')
    assert qutils._might_be_rich_text_heuristic(text) == bool(func(text))


def test_checks_of_this_module():
    assert tooltip_problem('Short\nlines') is None
    assert tooltip_problem(LONG_LINE)[0] == len(LONG_LINE)
    assert 'HTML' in tooltip_problem('Title\nSome <b>bold</b> text')[1]
    assert tooltip_problem('phase_<name> columns, area < 500') is None
    assert tooltip_problem('Condition: x < a & y > 1') is None
    assert tooltip_problem(f'<b>Title</b> {LONG_LINE}') is None
    assert tooltip_problem(f'<pre>{LONG_LINE}</pre>')[0] == len(LONG_LINE)
    assert tooltip_problem(f'<pre>{qutils.wrap_tooltip(LONG_LINE)}</pre>') is None
    assert tooltip_problem(f'<nobr>{LONG_LINE}</nobr>') is not None
    assert tooltip_problem(
        f'<nobr>{LONG_LINE[:70]}</nobr><br><nobr>{LONG_LINE[:70]}</nobr>'
    ) is None


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_collector_finds_every_kind_of_tooltip(app):
    """Widget, action, item model and plot item tool tips and point hover
    tips are all collected (otherwise the tests below would pass
    vacuously)."""
    from qtpy.QtWidgets import (
        QDialog, QListWidget, QMenu, QPushButton, QToolBar, QVBoxLayout
    )
    dialog = QDialog()
    dialog.setWindowTitle('Kinds')
    layout = QVBoxLayout(dialog)
    button = QPushButton('Button')
    button.setToolTip(f'button {LONG_LINE}')
    layout.addWidget(button)
    toolbar = QToolBar()
    action = QAction('Action', dialog)
    action.setToolTip(f'action {LONG_LINE}')
    toolbar.addAction(action)
    layout.addWidget(toolbar)
    # Action without parent in a menu without parent (added to a widget)
    menu = QMenu('Menu')
    menuAction = QAction('Menu action')
    menuAction.setToolTip(f'menu {LONG_LINE}')
    menu.addAction(menuAction)
    toolbar.addAction(menu.menuAction())
    listWidget = QListWidget()
    listWidget.addItem('Item')
    listWidget.item(0).setToolTip(f'item {LONG_LINE}')
    layout.addWidget(listWidget)
    plot = pg.PlotWidget()
    line = pg.InfiniteLine(1)
    line.setToolTip(f'line {LONG_LINE}')
    plot.addItem(line)
    scatter = pg.ScatterPlotItem(
        x=[0], y=[0], data=[LONG_LINE], hoverable=True,
        tip=lambda x, y, data: f'point {data}'
    )
    plot.addItem(scatter)
    layout.addWidget(plot)
    try:
        offenders = '\n'.join(check(collect_tooltips(dialog)))
        for kind in ('button', 'action', 'menu', 'item', 'line', 'point'):
            assert f"'{kind} word0" in offenders, kind
        assert "QDialog \"Kinds\": QPushButton \"Button\"" in offenders
    finally:
        dialog.deleteLater()


# --- Viewer -----------------------------------------------------------------

def fucci_template():
    path = os.path.join(PREMADE_CONFIGURATIONS_PATH, 'fucci_cell_cycle.json')
    with open(path, encoding='utf-8') as file:
        return json.load(file)


@pytest.fixture
def fucci():
    """FUCCI premade configuration resolved as when it is loaded."""
    return MeasurementsViewerWin.resolvePremadeTemplate(
        fucci_template(), {'GEMININ': GFP, 'CDT1': MCHERRY}
    )


def synthetic_table():
    frames, IDs = np.arange(4), [1, 2]
    index = pd.MultiIndex.from_product(
        [frames, IDs], names=['frame_i', 'Cell_ID']
    )
    df = pd.DataFrame(index=index).sort_index(level='Cell_ID')
    n = len(df)
    df['area'] = [10., 20., 15., 30., 5., 6., 7., 4.]
    df['intensity'] = [100., 250., 150., 300., 90., 95., 99., 98.]
    df['cell_vol_fl'] = np.linspace(30, 60, n)
    for ch, values in ((GFP, [.1, .3, .6, .9]), (MCHERRY, [.9, .6, .3, .1])):
        df[f'{ch}_fucci'] = values * 2
        df[f'{ch}_ramp'] = [0, 1, 1, -1] * 2
    df['fucci_crossing'] = [0, 0, 1, 0] * 2
    df['fucci_rel_diff'] = np.linspace(-1, 1, n)
    # Lineage tree: other-cell sources (e.g. daughter) are available
    df['parent_ID_tree'] = [0, 1] * 4
    marked = {'G1_S': 1, 'Crossing': 2, 'G2_start': 3}
    for name in ('G1_S', 'Cdt1_peak', 'Crossing', 'G2_start', 'M_start'):
        df[f'timepoint_{name}'] = [
            frame == marked.get(name) for frame in frames
        ] * 2
    return df.sort_index()


@pytest.fixture
def viewer(app, monkeypatch):
    """Viewer with a small synthetic table (as in the other viewer tests)."""
    monkeypatch.setattr(MeasurementsViewerWin, 'loadConfiguration', Mock())
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    labels = np.zeros((4, 30, 30), dtype=np.uint16)
    labels[:, 2:5, 3:7] = 1
    labels[:, 15:19, 20:25] = 2
    acdc_df = synthetic_table()
    window.data = [SimpleNamespace(
        SizeT=4, SizeZ=1, frame_i=0, segm_data=labels,
        chNames=[GFP, MCHERRY], pos_path=r'C:\experiment\Position_1',
        acdc_df=acdc_df
    )]
    window.isSegm3D = False
    window.acdc_df = acdc_df
    monkeypatch.setattr(window, 'updateImages', Mock())
    monkeypatch.setattr(window, 'setViewRangeToObjBbox', Mock())
    monkeypatch.setattr(window, 'measurementProcessingWarning', Mock())
    monkeypatch.setattr(window, 'drawCustomAnnotationIDs', Mock())
    window.centralWidget().setEnabled(True)
    window.initFramesNavigation()
    window.initIDsNavigation(ID=1)
    window.updateColumnsList()
    window.plotMeasurements()
    yield window
    window.reinitCustomAnnot()
    window.deleteLater()
    app.processEvents()


def set_plotted(viewer, names):
    for item in viewer.columnsModel.measurementItems():
        item.setCheckState(
            Qt.Checked if item.text() in names else Qt.Unchecked
        )


def setup_dynamic_tooltips(viewer, fucci):
    """Time point identifiers, custom annotations, conditional formatting
    and phases of the FUCCI configuration, plus own ones with a long
    predictor, a long description and a long condition."""
    for values in fucci['timepoint_identifiers']:
        viewer.addTimepointIdentifierItem(TimepointIdentifier(**values))
    # 'S' clashes with Shift+S (shuffle colormap): note in the tool tip
    viewer.addTimepointIdentifierItem(TimepointIdentifier(
        'Bud emergence of the first daughter', (0, 0, 255), 'S',
        {'type': 'equation', 'equation': LONG_EQUATION,
         'mode': 'rising_true'}
    ))
    # Shortcut with a modifier (longer "Shortcut: ..., Auto: ..." line)
    viewer.addTimepointIdentifierItem(TimepointIdentifier(
        'Septum', (0, 200, 200), 'Ctrl+J'
    ))
    annotation = {
        'type': 'Single time-point', 'symbol': "'o' circle",
        'symbolColor': [0, 0, 255, 255], 'description': '',
        'keepActive': True, 'isHideChecked': True,
    }
    own = [
        {
            **annotation, 'name': 'Stopped growing',
            # Used by the "Cdt1_peak" time point: note in the tool tip
            'shortcut': 'Q', 'description': LONG_DESCRIPTION,
            'autoApply': True, 'overridesPhases': True,
            'predictor': {'type': 'equation', 'equation': LONG_EQUATION},
        },
        # Shortcut with modifiers
        {**annotation, 'name': 'Dividing', 'shortcut': 'Ctrl+Alt+J'},
    ]
    viewer.setCustomAnnotations([*fucci['custom_annotations'], *own])
    viewer.conditionalFormatting = {
        **fucci['conditional_formatting'],
        'area': [{'condition': LONG_CONDITION, 'color': '#0072b2'}],
    }
    viewer.phaseTranslatorRules = fucci['phase_translator_rules']
    viewer.showPhasesCheckbox.setChecked(True)
    assert viewer.applyPhaseTranslator(quiet=True)
    set_plotted(viewer, ['area', f'{GFP}_fucci'])
    viewer.plotMeasurements()


DIALOG_CLASSES = [
    apps.CombineMetricsEquationDialog,
    apps.customAnnotationDialog,
    apps.FunctionParamsDialog,
    apps_measurements_viewer.TimepointIdentifierDialog,
    apps_measurements_viewer.TimepointPredictorEquationDialog,
    apps_measurements_viewer.CustomAnnotationPredictorEquationDialog,
    apps_measurements_viewer._TimepointCellFilterEquationDialog,
    apps_measurements_viewer._FormattingConditionDialog,
    apps_measurements_viewer.ConditionalFormattingDialog,
    apps_measurements_viewer.PhaseTranslatorDialog,
    apps_measurements_viewer.PhaseTranslatorRuleDialog,
    apps_measurements_viewer.ChannelPlaceholderMappingDialog,
    apps_measurements_viewer.MeasurementProcessingRecipeDialog,
]


@pytest.fixture
def opened(monkeypatch):
    """Replace ``exec_`` of the viewer dialogs: collect the tool tips of the
    dialog when it would be shown, then return as if it was cancelled.
    Returns ``(opened dialogs, {tool tip: owner})``."""
    dialogs, tooltips = [], {}

    def exec_(self, *args, **kwargs):
        dialogs.append(self)
        collect_tooltips(self, tooltips)
        self.cancel = True
        return 0

    for cls in DIALOG_CLASSES:
        monkeypatch.setattr(cls, 'exec_', exec_)
    return dialogs, tooltips


def last_opened(dialogs, cls):
    dialog = dialogs[-1]
    assert type(dialog) is cls, dialogs
    return dialog


def open_viewer_dialogs(viewer, dialogs):
    """Open (and nest) the dialogs of the viewer as the user would."""
    mv = apps_measurements_viewer

    viewer.combineMeasurements()
    last_opened(dialogs, apps.CombineMetricsEquationDialog)

    viewer.processMeasurements()
    recipe = last_opened(dialogs, mv.MeasurementProcessingRecipeDialog)
    selector = recipe.preProcessParamsWidget.stepsWidgets[1]['selector']
    for i in range(selector.count()):
        selector.setCurrentIndex(i)
        # Parameters dialog with the info of the run() docstring
        selector.askSetParams()
        last_opened(dialogs, apps.FunctionParamsDialog)

    # Time point identifiers: new and edited (predictor editor, filters)
    viewer.addTimepointIdentifier()
    last_opened(dialogs, mv.TimepointIdentifierDialog)
    identifier, = [
        identifier for identifier in viewer.timepointIdentifiers
        if identifier.name == 'M_start'  # with other-cell settings
    ]
    viewer.execTimepointIdentifierDialog(
        identifier, 'Edit time point identifier'
    )
    identifierDialog = last_opened(dialogs, mv.TimepointIdentifierDialog)
    identifierDialog.editEquation()
    equationDialog = last_opened(dialogs, mv.TimepointPredictorEquationDialog)
    equationDialog.editMainCellCondition()
    last_opened(dialogs, mv._TimepointCellFilterEquationDialog)
    equationDialog.editOtherCellsCondition()
    last_opened(dialogs, mv._TimepointCellFilterEquationDialog)

    # Custom annotations: new and edited (predictor editor)
    viewer.addCustomAnnotation()
    last_opened(dialogs, apps.customAnnotationDialog)
    viewer.customAnnotModify(viewer.customAnnotButtonByName('Arrested'))
    annotDialog = last_opened(dialogs, apps.customAnnotationDialog)
    annotDialog.predictorGroupbox.editEquation()
    last_opened(dialogs, mv.CustomAnnotationPredictorEquationDialog)

    # Phases: translator, new rule and rule with eligibility formula
    viewer.configurePhaseTranslator()
    phaseDialog = last_opened(dialogs, mv.PhaseTranslatorDialog)
    phaseDialog.addRule()
    last_opened(dialogs, mv.PhaseTranslatorRuleDialog)
    phaseDialog.rules[0]['predictor'] = {
        'type': 'equation', 'equation': LONG_EQUATION
    }
    phaseDialog.rulesList.setCurrentRow(0)
    phaseDialog.editRule()
    ruleDialog = last_opened(dialogs, mv.PhaseTranslatorRuleDialog)
    ruleDialog.predictorGroupbox.editEquation()
    last_opened(dialogs, mv.CustomAnnotationPredictorEquationDialog)

    # Conditional formatting and its condition editor
    viewer.editConditionalFormatting('area')
    last_opened(dialogs, mv.ConditionalFormattingDialog)
    viewer.editFormattingCondition(LONG_CONDITION)
    last_opened(dialogs, mv._FormattingConditionDialog)


def test_viewer_window_tooltips(viewer, fucci):
    setup_dynamic_tooltips(viewer, fucci)
    tooltips = collect_tooltips(viewer)
    # Comparison mode (measurement vs measurement)
    viewer.measurementsModeCombobox.setCurrentIndex(1)
    set_plotted(viewer, ['area', 'intensity'])
    assert viewer.measurementsPlot.listDataItems()
    collect_tooltips(viewer, tooltips)
    owners ='\n'.join(tooltips.values())
    texts = '\n'.join(tooltips)
    # The dynamic tool tips are covered
    assert 'hover tip of a point' in owners
    assert 'Formatting:' in texts and 'S Late: frames' in texts
    assert 'Stopped growing' in texts and 'downstream' in texts
    assert 'Bud emergence of the first daughter' in texts
    assert_no_offenders(tooltips)


def test_viewer_dialog_tooltips(viewer, fucci, opened):
    dialogs, tooltips = opened
    setup_dynamic_tooltips(viewer, fucci)
    open_viewer_dialogs(viewer, dialogs)
    assert_no_offenders(tooltips)


@pytest.mark.parametrize('filename', sorted(
    filename for filename in os.listdir(PREMADE_CONFIGURATIONS_PATH)
    if filename.endswith('.json')
))
def test_premade_setup_dialog_tooltips(app, filename):
    """Channel and input setup dialog of each premade configuration (with
    the background values section, as the viewer opens it)."""
    path = os.path.join(PREMADE_CONFIGURATIONS_PATH, filename)
    with open(path, encoding='utf-8') as file:
        template = json.load(file)
    placeholders = template['channel_placeholders']
    inputs = template.get('input_placeholders')
    backgroundValues = None
    if 'background_values' in template:
        backgroundValues = MeasurementsViewerWin.validateBackgroundValuesSpec(
            template['background_values'], placeholders, inputs
        )
    dialog = apps_measurements_viewer.ChannelPlaceholderMappingDialog(
        placeholders, [GFP, MCHERRY, 'phase_contr'], inputs=inputs,
        validate=Mock(), backgroundValues=backgroundValues,
        computeBackgroundValues=Mock(), tableStartDir='',
        channelHints=template.get('channel_hints')
    )
    try:
        assert_no_offenders(collect_tooltips(dialog))
    finally:
        dialog.deleteLater()
