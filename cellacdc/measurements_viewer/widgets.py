"""Private widgets and pyqtgraph items of the measurements viewer: the time
course plot item and the view boxes of its extra y axes, time point
lines, the columns list delegate, the shortcut-aware action and
auto-shortcut text, stand-ins for main-GUI objects, and the measurements
tree model of the columns panel and the processing recipe dialog.
"""
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel, QStandardItem
from cellacdc import measurements
from cellacdc.shared.widgets import search_widgets as widgets_search_widgets


class MeasurementTreeModel(QStandardItemModel):
    """Shared measurement hierarchy with selectable/checkable leaves only."""

    def populate(self, dataframe, channels=(), checkable=False):
        self.clear()
        nodes = {(): self.invisibleRootItem()}
        for path, columns in measurements.measurement_column_groups(dataframe, channels):
            for depth in range(1, len(path)+1):
                prefix = path[:depth]
                if prefix not in nodes:
                    group = QStandardItem(prefix[-1])
                    group.setFlags(Qt.ItemIsEnabled)
                    nodes[prefix[:-1]].appendRow(group)
                    nodes[prefix] = group
            for column in columns:
                item = QStandardItem(column)
                item.setEditable(False)
                item.setCheckable(checkable)
                item.setData(column, widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE)
                nodes[path].appendRow(item)

    def measurementItems(self, parent=None):
        if parent is None:
            parent = self.invisibleRootItem()
        for row in range(parent.rowCount()):
            item = parent.child(row)
            if item.hasChildren():
                yield from self.measurementItems(item)
            else:
                yield item


# Old names of cellacdc.widgets that code reached through this module
# path before it existed (cellacdc/_compat.py).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.forward_module(
    __name__, 'cellacdc.widgets')
