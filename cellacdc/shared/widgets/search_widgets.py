"""Search widgets: ButtonSearchWidget (finds the buttons, actions and
measurements of a window), the fuzzy-matching proxy models
ButtonSearchCompleter and MeasurementTreeSearchProxy, the highlight
delegate _ButtonSearchHighlightDelegate (subclassed by the measurements
viewer) and the text matching helpers (_fuzzySearchSimilarity,
_searchSynonyms, _matchesSearchText, _normalizeSearchText,
_searchHighlightRanges, _highlightSearchText). Leaf module: it imports
no other widget module.
"""
import re
from difflib import SequenceMatcher
import html
import natsort
from qtpy.QtCore import Signal, Qt, QEvent, QSortFilterProxyModel
from qtpy.QtGui import (
    QPalette, QStandardItemModel, QStandardItem, QTextDocument
)
from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QAction, QStyle, QListView,
    QAbstractItemView, QStyledItemDelegate, QStyleOptionViewItem
)
from cellacdc import html_utils
from cellacdc import rst_utils


class ButtonSearchCompleter(QSortFilterProxyModel):
    """Filter model rows using configurable name and tooltip roles."""

    SEARCH_NAME_ROLE = Qt.UserRole + 3
    SYNONYMS_ROLE = Qt.UserRole + 4
    PRIORITY_ROLE = Qt.UserRole + 5
    SECONDARY_NAME_ROLE = Qt.UserRole + 1
    TOOLTIP_ROLE = Qt.UserRole

    def __init__(self, parent=None, searchRoles=None, tooltipRole=TOOLTIP_ROLE):
        super().__init__(parent)
        if searchRoles is None:
            searchRoles = (
                self.SEARCH_NAME_ROLE,
                self.SECONDARY_NAME_ROLE,
                self.SYNONYMS_ROLE,
            )
        self.searchRoles = tuple(searchRoles)
        self.tooltipRole = tooltipRole
        self._filterText = ''
        self.FUZZY_THRESHOLD = 0.5
        self._scoreCache = {}
        self._similarityCache = {}

    def setFilterText(self, text):
        self._filterText = text.strip().casefold()
        self.refresh()

    def refresh(self):
        """Clear query-local caches and re-apply filtering and sorting."""
        self._scoreCache.clear()
        self._similarityCache.clear()
        # self.invalidateFilter()
        self.invalidate()
        
    def _getScore(self, index):
        cache_key = (index.row(), self._filterText)
        score = self._scoreCache.get(cache_key)
        if score is None:
            score = self._matchScore(index)
            self._scoreCache[cache_key] = score
        return score

    def lessThan(self, left, right):
        left_score = self._getScore(left)
        right_score = self._getScore(right)
        
        primary_left, secondary_left = left_score
        primary_right, secondary_right = right_score
        
        if primary_left != primary_right:
            return primary_left < primary_right
        
        if secondary_left != secondary_right:
            return secondary_left < secondary_right
        
        return left.row() < right.row()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._filterText:
            return True
        
        index = self.sourceModel().index(source_row, 0, source_parent)
        score = self._getScore(index)
        return score[0] < 7
    
    def _matchScore(self, index):
        if index.data(self.PRIORITY_ROLE):
            return -1, 0

        names = []
        for role in self.searchRoles:
            value = index.data(role)
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            names.extend(
                str(name).strip().casefold()
                for name in values
                if name is not None and str(name).strip()
            )

        if any(name == self._filterText for name in names):
            return 0, 0
        if any(name.startswith(self._filterText) for name in names):
            return 1, 0
        if any(self._filterText in name for name in names):
            return 2, 0

        best_similarity = max((self._similarity(name) for name in names), default=0.0)
        if best_similarity >= self.FUZZY_THRESHOLD:
            return 3, -best_similarity

        tooltip = ''
        if self.tooltipRole is not None:
            tooltip = index.data(self.tooltipRole) or ''
            tooltip = str(tooltip).casefold()
        if tooltip.startswith(self._filterText):
            return 4, 0
        if self._filterText in tooltip:
            return 5, 0

        tooltip_similarity = self._similarity(tooltip)
        if tooltip_similarity >= self.FUZZY_THRESHOLD:
            return 6, -tooltip_similarity

        return 7, 0

    def _similarity(self, text):
        cache_key = (text, self.FUZZY_THRESHOLD)
        similarity = self._similarityCache.get(cache_key)
        if similarity is None:
            similarity = _fuzzySearchSimilarity(
                self._filterText, text, self.FUZZY_THRESHOLD
            )
            self._similarityCache[cache_key] = similarity
        return similarity


class MeasurementTreeSearchProxy(ButtonSearchCompleter):
    """Keep ancestors of fuzzy-matched measurement leaves visible."""

    def __init__(self, parent=None):
        super().__init__(
            parent, searchRoles=(self.SEARCH_NAME_ROLE,), tooltipRole=None
        )
        self.setRecursiveFilteringEnabled(True)
        self.setDynamicSortFilter(True)
        self.sort(0, Qt.AscendingOrder)

    def _getScore(self, index):
        # Sibling row numbers repeat across tree branches.
        key = (index, self._filterText)
        if key not in self._scoreCache:
            self._scoreCache[key] = self._matchScore(index)
        return self._scoreCache[key]

    def filterAcceptsRow(self, source_row, source_parent):
        index = self.sourceModel().index(source_row, 0, source_parent)
        if self._filterText and self.sourceModel().hasChildren(index):
            return False
        return super().filterAcceptsRow(source_row, source_parent)

    def lessThan(self, left, right):
        if self.sourceModel().hasChildren(left) or self.sourceModel().hasChildren(right):
            return left.row() < right.row()
        return super().lessThan(left, right)


def _fuzzySearchSimilarity(query, text, minimum_similarity=0.0):
    """Return the best partial ratio, or zero below the requested minimum."""
    if not query or not text:
        return 0.0

    query_length = len(query)
    candidate_length = min(query_length, len(text))
    if (
            2.0 * candidate_length / (query_length + candidate_length)
            < minimum_similarity
        ):
        return 0.0

    matcher = SequenceMatcher(None, query, text)
    blocks = matcher.get_matching_blocks()
    best_ratio = 0.0
    seen_starts = set()
    for query_start, text_start, _ in blocks:
        start = max(text_start - query_start, 0)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        candidate = text[start:start + query_length]
        matcher.set_seq2(candidate)
        # These upper bounds cannot discard a qualifying or better match.
        lower_bound = max(minimum_similarity, best_ratio)
        if matcher.real_quick_ratio() < lower_bound:
            continue
        if matcher.quick_ratio() < lower_bound:
            continue
        ratio = matcher.ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio == 1.0:
            break
    return best_ratio if best_ratio >= minimum_similarity else 0.0


def _searchSynonyms():
    synonyms = [
        ('Segm.', 'Segmentation'),
        ('Edit', 'Change'),
        ('ID', 'IDs'),
        ('ID', 'cell'),
        ('ID', 'object'),
        ('bkgr', 'background'),
    ]
    return synonyms + [(synonym, phrase) for phrase, synonym in synonyms]


def _matchesSearchText(query, text):
    """Match checkbox labels literally, including synonym substitutions."""
    query = query.strip().casefold()
    if not query:
        return False

    text = text.casefold()
    query_variants = [query]
    for phrase, synonym in _searchSynonyms():
        phrase = phrase.casefold()
        if phrase in query:
            query_variants.append(query.replace(phrase, synonym.casefold()))

    for query_variant in query_variants:
        if query_variant in text:
            return True
    return False


def _normalizeSearchText(text):
    """Case-fold text and remove non-alphanumeric search separators."""
    return ''.join(
        character.casefold()
        for character in text
        if character.isalnum()
    )


def _searchHighlightRanges(query, text, threshold=0.5):
    """Return display-text ranges for exact or fuzzy search matches."""
    normalized_query = _normalizeSearchText(query)
    if not normalized_query:
        return []

    normalized_text = []
    source_positions = []
    for position, character in enumerate(text):
        if not character.isalnum():
            continue
        folded = character.casefold()
        normalized_text.extend(folded)
        source_positions.extend([position] * len(folded))
    normalized_text = ''.join(normalized_text)
    if not normalized_text:
        return []

    def source_ranges(start, end):
        ranges = []
        for position in source_positions[start:end]:
            if ranges and position <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], position + 1)
            else:
                ranges.append((position, position + 1))
        return ranges

    match_start = normalized_text.find(normalized_query)
    if match_start >= 0:
        return source_ranges(
            match_start, match_start + len(normalized_query)
        )

    matcher = SequenceMatcher(None, normalized_query, normalized_text)
    best_ratio = threshold
    best_start = None
    for query_start, text_start, _ in matcher.get_matching_blocks():
        candidate_start = max(text_start - query_start, 0)
        candidate = normalized_text[
            candidate_start:candidate_start + len(normalized_query)
        ]
        ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
        if ratio >= best_ratio and (
                best_start is None or ratio > best_ratio
            ):
            best_ratio = ratio
            best_start = candidate_start

    if best_start is None:
        return []

    candidate = normalized_text[
        best_start:best_start + len(normalized_query)
    ]
    normalized_ranges = []
    for block in SequenceMatcher(
            None, normalized_query, candidate
        ).get_matching_blocks():
        if not block.size:
            continue
        start = best_start + block.b
        end = start + block.size
        if normalized_ranges and start <= normalized_ranges[-1][1]:
            normalized_ranges[-1] = (
                normalized_ranges[-1][0],
                max(normalized_ranges[-1][1], end),
            )
        else:
            normalized_ranges.append((start, end))

    ranges = []
    for start, end in normalized_ranges:
        ranges.extend(source_ranges(start, end))

    merged_ranges = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (
                merged_ranges[-1][0],
                max(merged_ranges[-1][1], end),
            )
        else:
            merged_ranges.append((start, end))
    return merged_ranges


def _highlightSearchText(text, ranges, color=None):
    """Return escaped rich text with the requested display-text ranges bolded."""
    parts = []
    previous_end = 0
    for start, end in ranges:
        style = 'font-weight:bold;'
        if color is not None:
            style = f'color:{color}; {style}'
        parts.extend((
            html.escape(text[previous_end:start]),
            (
                f'<span style="{style}">'
                f'{html.escape(text[start:end])}</span>'
            ),
        ))
        previous_end = end
    parts.append(html.escape(text[previous_end:]))
    return ''.join(parts)


class _ButtonSearchHighlightDelegate(QStyledItemDelegate):
    """Paint search matches in list entries using the dialog's highlighting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filterText = ''

    def setFilterText(self, text):
        self._filterText = text.strip()

    def _highlightRanges(self, text, threshold):
        return _searchHighlightRanges(
            self._filterText, text, threshold
        )

    def paint(self, painter, option, index):
        text = index.data(Qt.DisplayRole) or ''
        text = str(text)
        proxy_model = index.model()
        threshold = getattr(proxy_model, 'FUZZY_THRESHOLD', 0.5)
        ranges = self._highlightRanges(text, threshold)
        if not ranges:
            return super().paint(painter, option, index)

        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        text_color_role = (
            QPalette.HighlightedText
            if item_option.state & QStyle.State_Selected
            else QPalette.Text
        )
        text_color = item_option.palette.color(text_color_role).name()

        item_option.text = ''
        style = (
            item_option.widget.style()
            if item_option.widget is not None
            else QApplication.style()
        )
        style.drawControl(
            QStyle.CE_ItemViewItem,
            item_option,
            painter,
            item_option.widget,
        )
        text_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText,
            item_option,
            item_option.widget,
        )
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(item_option.font)
        document.setHtml(
            f'<span style="color:{text_color};">'
            f'{_highlightSearchText(text, ranges)}</span>'
        )
        document.setTextWidth(text_rect.width())

        painter.save()
        vertical_offset = max(
            (text_rect.height() - document.size().height()) / 2,
            0,
        )
        painter.translate(
            text_rect.left(), text_rect.top() + vertical_offset
        )
        painter.setClipRect(
            0, 0, text_rect.width(), text_rect.height()
        )
        document.drawContents(painter)
        painter.restore()


class ButtonSearchWidget(QWidget):
    """Search widget populated from documentation or caller-provided records.

    Parameters
    ----------
    guiWin:
        Optional GUI object used by :meth:`registerSearchControls`.
    loadingType:
        ``'rst'`` loads button metadata from the documentation. ``None``
        starts with no records. Ignored when ``searchRecords`` is provided.
    searchRecords:
        Optional mapping of record keys to dictionaries. Each dictionary may
        contain ``display``, ``search_name``, ``synonyms``, ``tooltip``,
        ``button_id``, and ``target``. All fields are optional: the mapping key
        is used as the default display and search name, while collections of
        strings may be supplied as synonyms. ``target`` may be a widget,
        action, GUI attribute name, or a list/tuple of these to highlight
        together. Toolbar controls also highlight the opener controls
        registered with :meth:`registerToolbarTargets`.
    """

    ACTION_ROLE = Qt.UserRole + 6
    ACTION_VALUE_ROLE = Qt.UserRole + 7

    sigTriggerBlink = Signal(object)
    sigSearchId = Signal(int)

    def __init__(
            self,
            guiWin=None,
            loadingType='rst',
            searchRecords=None,
        ):
        super().__init__()
        self.guiWin = guiWin

        self.buttons_data = {}
        self._toolbar_targets = {}

        self.init_ui(loadingType=loadingType, searchRecords=searchRecords)

    def _loadSearchRecordsFromRst(self):
        buttons = rst_utils.get_tooltips_from_docs(for_search=True)
        records = {}
        for button in buttons:
            name = button['name']
            button_id = button.get('id')
            normalized_name = name.strip().casefold()
            display = f'{name} ({button_id})' if button_id else name
            records[normalized_name] = {
                'display': display,
                'search_name': normalized_name,
                'synonyms': self._synonymsForName(name),
                'tooltip': button.get('tooltip', ''),
                'button_id': button_id,
                'target': None,
            }
            if button_id is not None:
                self.buttons_data[button_id] = button
        return records

    def _loadSearchRecords(self, loadingType, searchRecords):
        if searchRecords is not None:
            records = searchRecords
        elif loadingType == 'rst':
            records = self._loadSearchRecordsFromRst()
        elif loadingType is None:
            records = {}
        else:
            raise ValueError(f'Unsupported search loading type: {loadingType}')

        normalized_records = {}
        for record_key, source_record in records.items():
            record = dict(source_record or {})
            search_name = str(
                record.get('search_name', record_key)
            ).strip().casefold()
            if not search_name:
                continue
            record['display'] = str(record.get('display', record_key))
            record['search_name'] = search_name
            record.setdefault('synonyms', [])
            record.setdefault('tooltip', '')
            record.setdefault('button_id', None)
            record.setdefault('target', None)
            normalized_records[search_name] = record
        return normalized_records

    def init_ui(self, loadingType='rst', searchRecords=None):
        self._search_records = self._loadSearchRecords(
            loadingType, searchRecords
        )

        self.model, self._items_by_name = self._build_source_model()
        self.proxy_model = self._build_proxy_model(self.model)
        self._controls_model = self.model
        self._controls_proxy_model = self.proxy_model
        self._IDs_model = None
        self._IDs_proxy_model = None
        self._IDs_in_model = None

        # Search field
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for functionality or IDs")
        self.search_input.setFocusPolicy(Qt.ClickFocus)
        # add stretch to make the search input expand
        self.search_input.setFixedWidth(250)
        self.search_input.installEventFilter(self)

        # Custom popup (replaces QCompleter so we can fully control
        # when search vs. navigation happen). Reparented to the actual
        # top-level window right before showing (see show_popup) instead
        # of a separate Qt.Popup window, so it never steals OS-level
        # keyboard focus away from the search field.
        self.popup = QListView()
        self.popup.setFocusPolicy(Qt.NoFocus)
        self.popup.setModel(self.proxy_model)
        self._highlight_delegate = _ButtonSearchHighlightDelegate(self.popup)
        self.popup.setItemDelegate(self._highlight_delegate)
        try:
            self.popup.setEditTriggers(QAbstractItemView.NoEditTriggers)
        except:
            pass
        try:
            self.popup.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        except:
            pass
        self.popup.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.popup.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.popup.setMouseTracking(True)
        self.popup.clicked.connect(self.on_popup_clicked)
        
        # Enable scrollbar when content exceeds max height
        self.popup.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.popup.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.popup.hide()

        # `textEdited` only fires on actual user keystrokes, not on
        # programmatic setText() (used by arrow-key navigation below),
        # so typing re-filters but navigating doesn't.
        self.search_input.textEdited.connect(
            self.on_search_text_changed
        )

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove all margins (left, top, right, bottom)
        layout.setSpacing(0)  # Remove spacing
        layout.addWidget(self.search_input)
        self.setLayout(layout)
        
        QApplication.instance().installEventFilter(self)

        
    def registerSearchControls(self):
        annotation_display_controls = (
            ('Contours', self.guiWin.annotContourCheckbox(0)),
            ('Segm. masks', self.guiWin.annotOverlaySegmMaskCheckbox(0)),
            ('IDs', self.guiWin.annotIDsCheckbox(0)),
            ('Lineage information', self.guiWin.annotLineageInfoCheckbox(0)),
            ('Cell cycle info', self.guiWin.annotCellCycleInfoCheckbox(0)),
            (
                'Mother-daughter line',
                self.guiWin.annotMotherDaughterLineCheckbox(0),
            ),
            ('Object tracks', self.guiWin.annotObjectTracksCheckbox(0)),
            ('Do not annotate', self.guiWin.annotDoNotAnntoateCheckbox(0)),
        )
        quick_settings_controls = (
            ('View pre-processed image', self.guiWin.viewPreprocDataToggle),
            ('View combined channels', self.guiWin.viewCombineChannelDataToggle),
            ('Autosave segmentation', self.guiWin.autoSaveToggle),
            ('Autosave annotations', self.guiWin.autoSaveAnnotToggle),
            ('Autosave interval', self.guiWin.autoSaveIntervalEditButton),
            ('Cell cycle annotation checker', self.guiWin.ccaIntegrCheckerToggle),
            ('Annotate lost objects', self.guiWin.annotLostObjsToggle),
            ('Show all contours', self.guiWin.showAllContoursToggle),
            ('Font size', self.guiWin.fontSizeSpinBox),
        )
        self.addItems(
            annotation_display_controls + quick_settings_controls
        )
        
    def synonyms(self):
        return _searchSynonyms()

    def registerToolbarTargets(self, toolbar, targets):
        """Link a toolbar's controls to the button(s) that open it."""
        self._toolbar_targets[toolbar] = (
            tuple(targets) if isinstance(targets, (list, tuple))
            else (targets,)
        )

    def highlightTargets(self, target):
        """Resolve a search target and include its toolbar opener controls."""
        targets = []

        def add_target(control):
            if isinstance(control, (list, tuple)):
                for member in control:
                    add_target(member)
                return
            if isinstance(control, str):
                control = getattr(self.guiWin, control)
            if not isinstance(control, (QWidget, QAction)):
                raise TypeError('Search targets must be widgets or actions')
            if control in targets:
                return
            targets.append(control)
            if isinstance(control, QAction):
                containers = (
                    control.associatedObjects()
                    if hasattr(control, 'associatedObjects')
                    else control.associatedWidgets()
                )
            else:
                containers = [control]
            for container in containers:
                if not isinstance(container, QWidget):
                    continue
                while container is not None:
                    for opener in self._toolbar_targets.get(container, ()):
                        add_target(opener)
                    container = container.parentWidget()

        add_target(target)
        return targets

    def _synonymsForName(self, name):
        normalized_name = name.strip().casefold()
        aliases = []
        for phrase, synonym in self.synonyms():
            normalized_phrase = phrase.strip().casefold()
            if normalized_phrase not in normalized_name:
                continue
            aliases.append(
                normalized_name.replace(
                    normalized_phrase, synonym.strip().casefold()
                )
            )
        return aliases

    def addItems(self, items):
        """Add ``(name, target)`` entries; targets may be grouped in a list/tuple."""
        for name, target in items:
            normalized_name = name.strip().casefold()
            if not normalized_name:
                continue

            controls = target if isinstance(target, (list, tuple)) else (target,)
            search_texts = []
            for control in controls:
                tooltip_getter = getattr(control, 'toolTip', None)
                tooltip = tooltip_getter() if callable(tooltip_getter) else ''
                search_text = html_utils.to_plain_text(tooltip).strip()
                shortcut_getter = getattr(control, 'shortcut', None)
                shortcut = (
                    shortcut_getter() if callable(shortcut_getter) else None
                )
                shortcut_text = shortcut.toString() if shortcut is not None else ''
                if shortcut_text and shortcut_text not in search_text:
                    search_text = (
                        f'{search_text}\nShortcut: {shortcut_text}'.strip()
                    )
                if search_text and search_text not in search_texts:
                    search_texts.append(search_text)
            search_text = '\n'.join(search_texts)

            record = self._search_records.get(normalized_name)
            if record is None:
                record = {
                    'display': name,
                    'search_name': normalized_name,
                    'synonyms': self._synonymsForName(name),
                    'tooltip': '',
                    'button_id': None,
                    'target': None,
                }
                self._search_records[normalized_name] = record
            elif search_text:
                documented_text = record['tooltip'] or ''
                search_text = f'{documented_text}\n{search_text}'.strip()

            if search_text:
                record['tooltip'] = search_text
            record['target'] = target

        self._rebuild_models()

    def _build_source_model(self):
        model = QStandardItemModel()
        items_by_name = {}
        records = natsort.natsorted(
            self._search_records.items(),
            key=lambda item: item[1]['display'],
        )
        for search_name, record in records:
            item = QStandardItem(record['display'])
            item.setData(
                record.get('tooltip', ''),
                ButtonSearchCompleter.TOOLTIP_ROLE
                )
            item.setData(
                record.get('button_id', None),
                ButtonSearchCompleter.SECONDARY_NAME_ROLE
            )
            item.setData(
                record.get('synonyms', []),
                ButtonSearchCompleter.SYNONYMS_ROLE,
            )
            item.setData(record.get('target', None), Qt.UserRole + 2)
            item.setData(
                search_name,
                ButtonSearchCompleter.SEARCH_NAME_ROLE,
            )
            model.appendRow(item)
            items_by_name[search_name] = item
            
        return model, items_by_name

    def _build_proxy_model(self, model):
        proxy_model = ButtonSearchCompleter()
        proxy_model.setSourceModel(model)
        proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy_model.setDynamicSortFilter(True)
        proxy_model.sort(0, Qt.AscendingOrder)
        return proxy_model

    def _rebuild_models(self):
        """Rebuild source and proxy models from explicit search records."""
        self.model, self._items_by_name = self._build_source_model()
        self.proxy_model = self._build_proxy_model(self.model)
        self._controls_model = self.model
        self._controls_proxy_model = self.proxy_model
        self.popup.setModel(self.proxy_model)
        self.proxy_model.setFilterText(self.search_input.text())

    def setSearchRecords(self, records):
        """Replace caller-provided records, preserving the current query."""
        self._search_records = self._loadSearchRecords(None, records)
        self._rebuild_models()
        self.on_search_text_changed(self.search_input.text())

    def set_height_based_on(self, reference_widget):
        """Set the height of the search input based on another widget's height."""
        height_curr = reference_widget.sizeHint().height()
        self.search_input.setFixedHeight(int(height_curr / 1.7))

    def on_search_text_changed(self, text):
        """Filter the model and update the popup."""
        query = text.strip()
        self._update_search_models(query)
        self._highlight_delegate.setFilterText(query)
        self.proxy_model.setFilterText(query)
        if not query or self.proxy_model.rowCount() == 0:
            self.popup.hide()
            return

        self.popup.setCurrentIndex(self.proxy_model.index(0, 0))
        self.show_popup()

    def _update_search_models(self, query):
        if self.guiWin is None or not query.isascii() or not query.isdecimal():
            if self.model is not self._controls_model:
                self.model = self._controls_model
                self.proxy_model = self._controls_proxy_model
                self.popup.setModel(self.proxy_model)
            return

        current_IDs = tuple(self._get_curr_IDs())
        if current_IDs != self._IDs_in_model:
            self._rebuild_IDs_model(current_IDs)

        item = self._IDs_model.item(0)
        item.setText(f'Search for ID {query}')
        item.setData(query, ButtonSearchCompleter.SEARCH_NAME_ROLE)
        item.setData(int(query), self.ACTION_VALUE_ROLE)

        self.model = self._IDs_model
        self.proxy_model = self._IDs_proxy_model
        self.popup.setModel(self.proxy_model)

    def _rebuild_IDs_model(self, current_IDs):
        model = QStandardItemModel()
        item = QStandardItem()
        item.setData(True, ButtonSearchCompleter.PRIORITY_ROLE)
        item.setData('search_id', self.ACTION_ROLE)
        model.appendRow(item)

        sorted_IDs = natsort.natsorted(
            current_IDs
        )
        for ID in sorted_IDs:
            item = QStandardItem(f'ID {ID}')
            item.setData(str(ID), ButtonSearchCompleter.SEARCH_NAME_ROLE)
            item.setData('search_id', self.ACTION_ROLE)
            item.setData(int(ID), self.ACTION_VALUE_ROLE)
            model.appendRow(item)

        self._IDs_model = model
        self._IDs_proxy_model = self._build_proxy_model(model)
        self._IDs_in_model = current_IDs

    def show_popup(self):
        """Position and show the popup below the search field."""

        # Reparent to the current top-level window each time: at
        # construction ButtonSearchWidget may not be embedded in its
        # final parent yet, so self.window() could still be itself,
        # clipping the popup to its own small bounds.
        window = self.window()
        if self.popup.parentWidget() is not window:
            self.popup.setParent(window)

        global_point = self.search_input.mapToGlobal(
            self.search_input.rect().bottomLeft()
        )
        point = window.mapFromGlobal(global_point)
        self.popup.move(point)
        self.popup.setFixedWidth(self.search_input.width())
 
        # Calculate height for maximum 5 visible rows
        row_height = self.popup.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 20  # Fallback default height
        
        # Show max 5 items, scrollbar appears if more results
        max_visible_rows = 5
        visible_rows = min(self.proxy_model.rowCount(), max_visible_rows)
        popup_height = row_height * visible_rows + 4  # +4 for spacing/borders
        
        self.popup.setMaximumHeight(popup_height)
        self.popup.setMinimumHeight(popup_height)
 
        self.popup.show()
        self.popup.raise_()

    def navigate(self, direction):
        """Move the popup selection without re-triggering the search."""

        if not self.popup.isVisible():
            return

        row_count = self.proxy_model.rowCount()
        if row_count == 0:
            return

        current_row = self.popup.currentIndex().row()
        new_row = min(max(current_row + direction, 0), row_count - 1)
        new_index = self.proxy_model.index(new_row, 0)

        self.popup.setCurrentIndex(new_index)
        self.popup.scrollTo(new_index)

    def on_popup_clicked(self, index):
        self.confirm_selection(index)

    def confirm_selection(self, index=None):
        """Emit sigTriggerBlink for the currently selected/matched entry."""

        if index is None or not index.isValid():
            index = self.popup.currentIndex() if self.popup.isVisible() else None

        if index is None or not index.isValid():
            # Fall back to the single remaining filtered result, if any.
            if self.proxy_model.rowCount() != 1:
                return
            index = self.proxy_model.index(0, 0)

        if index.data(self.ACTION_ROLE) == 'search_id':
            self.sigSearchId.emit(index.data(self.ACTION_VALUE_ROLE))
            self.search_input.clear()
            self.popup.hide()
            self.search_input.clearFocus()
            return

        target = index.data(Qt.UserRole + 2)
        if target is not None:
            self.search_input.clear()
            self.sigTriggerBlink.emit(target)
        else:
            text = index.data(Qt.DisplayRole)
            self.on_button_selected(text)
        self.popup.hide()
        self.search_input.clearFocus()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            try:
                global_pos = event.globalPosition().toPoint()
            except AttributeError:
                global_pos = event.globalPos()

            clicked_search = self.search_input.rect().contains(
                self.search_input.mapFromGlobal(global_pos)
            )
            clicked_popup = self.popup.isVisible() and self.popup.rect().contains(
                self.popup.mapFromGlobal(global_pos)
            )
            if not clicked_search and not clicked_popup:
                self.popup.hide()
                self.search_input.clearFocus()

        if obj is self.search_input and event.type() == QEvent.KeyPress:
            key = event.key()
            popup_visible = self.popup.isVisible()
            if key == Qt.Key_Down and popup_visible:
                self.navigate(1)
                return True
            elif key == Qt.Key_Up and popup_visible:
                self.navigate(-1)
                return True
            elif key in (Qt.Key_Return, Qt.Key_Enter) and popup_visible:
                self.confirm_selection()
                return True
            elif key == Qt.Key_Escape:
                if popup_visible:
                    self.popup.hide()
                self.search_input.clearFocus()
                return True

        elif obj is self.search_input and event.type() == QEvent.FocusOut:
            # The popup is a plain child widget (not an auto-dismissing
            # Qt.Popup), so hide it manually once the field is no longer
            # focused.
            self.popup.hide()
        elif obj is self.search_input and event.type() == QEvent.FocusIn:
            # Show the popup when the search input gains focus
            curr_text = self.search_input.text()
            self.on_search_text_changed(curr_text)

        return super().eventFilter(obj, event)

    def on_button_selected(self, text):
        """Handle when user selects a button from dropdown."""

        if not text:
            text = self.search_input.text()

        # Format:
        # "Button Name (buttonId)"
        #
        # Extract the ID from the final parentheses.
        match = re.search(r'\((\w+)\)$', text)

        if match:
            button_id = match.group(1)

            self.search_input.clear()

            self.sigTriggerBlink.emit(button_id)
            
            
    def _get_curr_IDs(self):
        guiWin = self.guiWin
        if guiWin is None or not getattr(guiWin, 'isDataLoaded', False):
            return ()
        posData = guiWin.data[guiWin.pos_i]
        IDs = posData.IDs
        return IDs
