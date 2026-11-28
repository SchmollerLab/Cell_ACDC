"""Combine-measurements equation editors (from apps.py).

_CombineMetricsDialog is also the base of the measurements viewer's
predictor, filter and conditional formatting editors, and
CombineMetricsEquationDialog is used by the main GUI, the Compute
measurements utility and the viewer. Also FormulaEditWidget (also used
to combine channels) and the unused CombineFeaturesCalculator. The
debug_trace decorator of timepoint_predictors.py is applied at
class-definition time. Keep equation-editor.rst in sync when changing
these editors.
"""
import re
import keyword
import numpy as np
import traceback
from functools import partial
import pandas as pd
import sympy as sp
from qtpy.QtGui import QIcon, QFont, QFont
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout,
    QGridLayout, QPlainTextEdit, QTreeWidget, QTreeWidgetItem, QTextEdit
)
from cellacdc import load, measurements, html_utils
from cellacdc import widgets
from cellacdc import fonts
from cellacdc.timepoint_predictors import debug_trace as timepoint_debug_trace
from . import base as dialogs_base
from ..widgets import style as widgets_style


class _CombineMetricsDialog(dialogs_base.QBaseDialog):
    # Rows of `acdc_df` used by "Test equation" for row-wise equations
    TEST_DATA_MAX_ROWS = 1000

    def __init__(
            self, allChNames, isZstack, isSegm3D, parent=None, debug=False,
            closeOnOk=True, acdcDfs=None, acdc_df=None,
            allowCellAggregations=False, limitToLoadedColumns=True
        ):
        super().__init__(parent)

        self.setWindowTitle('Add combined measurement')

        self.initAttributes()

        self.allChNames = allChNames
        self.acdcDfs = acdcDfs
        self.acdcDf = acdc_df
        # See `populateMeasurementCatalog`
        self.limitToLoadedColumns = limitToLoadedColumns
        self.allowCellAggregations = allowCellAggregations
        self.variableNames = []
        # Names added by `addTreeItems` (fast duplicate check)
        self._listedVariables = set()
        self.channelLessVariables = set()
        self.metricsTreeWidgets = []

        self.cancel = True
        self.closeOnOk = closeOnOk

        mainLayout = QVBoxLayout()
        equationLayout = QGridLayout()

        self.isZstack = isZstack
        self.isSegm3D = isSegm3D
        treesLayout = self.createMetricsTrees(isZstack, isSegm3D)
        self.measurementsSearchLineEdit = QLineEdit()
        self.measurementsSearchLineEdit.setPlaceholderText('Search measurement...')
        self.measurementsSearchLineEdit.setClearButtonEnabled(True)
        self.metricsSearchProxies = []
        self.metricsHighlightDelegates = []
        self._metricsSearchState = None
        for tree in self.metricsTreeWidgets:
            proxy = widgets.MeasurementTreeSearchProxy(self)
            proxy.setSourceModel(tree.model())
            self.metricsSearchProxies.append(proxy)
            delegate = widgets._ButtonSearchHighlightDelegate(tree)
            tree.setItemDelegate(delegate)
            self.metricsHighlightDelegates.append(delegate)
        self.measurementsSearchLineEdit.textChanged.connect(self.filterMeasurements)

        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24
        self.operatorButtons = []
        self.operators = [
            ('add', '+'), ('subtract', '-'), ('multiply', '*'),
            ('divide', '/'), ('open_bracket', '('), ('close_bracket', ')'),
            ('square', '**2'), ('pow', '**'), ('ln', 'log('),
            ('log10', 'log10('),
        ]
        for name, text in self.operators:
            button = widgets.PushButton(icon=QIcon(f':{name}.svg'))
            button.setIconSize(QSize(iconSize, iconSize))
            button.setToolTip(text)
            button.clicked.connect(partial(self.insertEquationText, text))
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)

        clearButton = widgets.PushButton(icon=QIcon(':clear.svg'))
        clearButton.setIconSize(QSize(iconSize, iconSize))
        clearEntryButton = widgets.PushButton(icon=QIcon(':backspace.svg'))
        clearEntryButton.setIconSize(QSize(iconSize, iconSize))
        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        categoricalOperatorsLayout = QHBoxLayout()
        categoricalOperatorsLayout.addWidget(QLabel('Comparisons / logic:'))
        self.categoricalOperatorButtons = []
        categoricalOperators = [
            ('==', ' == ', 'Equal to'),
            ('!=', ' != ', 'Not equal to'),
            ('<', ' < ', 'Less than (numerical or ordered inputs)'),
            ('<=', ' <= ', 'Less than or equal to (numerical or ordered inputs)'),
            ('>', ' > ', 'Greater than (numerical or ordered inputs)'),
            ('>=', ' >= ', 'Greater than or equal to (numerical or ordered inputs)'),
            ('AND', ' & ', 'Both conditions must be true: (condition1) & (condition2)'),
            ('OR', ' | ', 'Either condition must be true: (condition1) | (condition2)'),
            ('NOT', '~', 'Negate a condition: ~(condition)'),
            ('True', 'True', 'Boolean true'),
            ('False', 'False', 'Boolean false'),
        ]
        if self.allowCellAggregations:
            categoricalOperators.extend((
                ('ANY', 'any(', 'True on every frame if the condition is true in at least one frame of the cell'),
                ('ALL', 'all(', 'True on every frame only if the condition is true in every frame of the cell'),
            ))
        for label, text, tooltip in categoricalOperators:
            button = widgets.PushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(partial(self.insertEquationText, text))
            categoricalOperatorsLayout.addWidget(button)
            self.categoricalOperatorButtons.append(button)
        categoricalOperatorsLayout.addStretch(1)

        nameAndEqualsWidget = QWidget()
        nameAndEqualsLayout = QHBoxLayout(nameAndEqualsWidget)
        nameAndEqualsLayout.setContentsMargins(0, 0, 0, 0)
        self.newColNameLineEdit = widgets.alphaNumericLineEdit()
        self.newColNameLineEdit.setAlignment(Qt.AlignCenter)
        nameAndEqualsLayout.addWidget(self.newColNameLineEdit)
        nameAndEqualsLayout.addWidget(QLabel(' = '))
        equationLayout.addWidget(QLabel('New measurement name:'), 0, 0)
        equationLayout.addWidget(nameAndEqualsWidget, 1, 0, Qt.AlignTop)

        equationLayout.addWidget(QLabel('Equation:'), 0, 1)
        self.formulaEditWidget = FormulaEditWidget(
            self.variableNames, parent=self, allow_empty=False,
            validator=self.validateEquation
        )
        self.formulaEditWidget._edit.setPlaceholderText(
            'e.g. area / 2 or cell_cycle_stage == "G1"'
        )
        self.equationDisplay = self.formulaEditWidget
        equationLayout.addWidget(self.formulaEditWidget, 1, 1, Qt.AlignTop)
        equationLayout.setColumnStretch(0, 1)
        equationLayout.setColumnStretch(1, 2)

        cellAggregationInstructions = (
            'Wrap a condition in <code>any(...)</code> or <code>all(...)</code> '
            'to check it across a cell\'s frames; the result is repeated on '
            'each frame of that cell.<br>'
            if self.allowCellAggregations else ''
        )
        instructions = html_utils.paragraph(f"""
            <b>Double-click</b> an available input to add it to the equation.
            Numerical inputs support arithmetic; categorical inputs support
            comparisons (e.g. <code>cell_cycle_stage == "G1"</code>).
            Combine conditions with <code>&amp;</code> or <code>|</code>.<br>
            {cellAggregationInstructions}
            The result is saved as a column with the new measurement name.
        """)
        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.infoPushButton('  Help...')
        testButton = widgets.calcPushButton('Test equation')
        self.okButton = widgets.okPushButton(' Ok ')
        self.okButton.setDisabled(True)
        self.statusLabel = QLabel()
        buttonsLayout.addWidget(self.statusLabel)
        buttonsLayout.addStretch(1)
        if debug:
            debugButton = widgets.PushButton('Debug')
            debugButton.clicked.connect(self._debug)
            buttonsLayout.addWidget(debugButton)
        for button in (cancelButton, helpButton, testButton, self.okButton):
            buttonsLayout.addWidget(button)
        self.testOutputDisplay = QTextEdit()
        self.testOutputDisplay.setReadOnly(True)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(self.measurementsSearchLineEdit)
        mainLayout.addLayout(treesLayout)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(categoricalOperatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addWidget(self.testOutputDisplay)
        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.formulaEditWidget.backspace)
        helpButton.clicked.connect(self.showHelp)
        self.okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)
        self.formulaEditWidget.sigFormulaChanged.connect(self.equationChanged)
        self.newColNameLineEdit.textChanged.connect(self.equationChanged)
        self.setLayout(mainLayout)
        self.setFont(fonts.font)
        self.setStyleSheet(widgets_style.TREEWIDGET_STYLESHEET)

    def filterMeasurements(self, text):
        """Use the recipe selector's recursive fuzzy search on every table."""
        def treeItems(parent):
            for row in range(parent.childCount()):
                item = parent.child(row)
                yield item
                yield from treeItems(item)

        searching = bool(text.strip())
        if searching and self._metricsSearchState is None:
            self._metricsSearchState = [
                (item, item.isHidden(), item.isExpanded())
                for tree in self.metricsTreeWidgets
                for item in treeItems(tree.invisibleRootItem())
            ]
        for tree, proxy, delegate in zip(
                self.metricsTreeWidgets, self.metricsSearchProxies,
                self.metricsHighlightDelegates
            ):
            proxy.setFilterText(text)
            delegate.setFilterText(text)
            if searching:
                for item in treeItems(tree.invisibleRootItem()):
                    matches = proxy.mapFromSource(tree.indexFromItem(item)).isValid()
                    item.setHidden(not matches)
                    if matches and item.childCount():
                        item.setExpanded(True)
            tree.viewport().update()
        if not searching and self._metricsSearchState is not None:
            for item, hidden, expanded in self._metricsSearchState:
                item.setHidden(hidden)
                item.setExpanded(expanded)
            self._metricsSearchState = None

    def createMetricsTrees(self, isZstack, isSegm3D):
        layout = QHBoxLayout()
        if self.acdcDfs is not None:
            for index, (name, df) in enumerate(self.acdcDfs.items()):
                tree = self.createMetricsTree()
                numerical = QTreeWidgetItem(tree, ['Numerical inputs'])
                categorical = QTreeWidgetItem(tree, ['Categorical inputs'])
                self.populateDataframeTree(numerical, categorical, df, index)
                tree.expandToDepth(0)
                treeLayout = QVBoxLayout()
                treeLayout.addWidget(QLabel(f'{index+1}. {name} measurements'))
                treeLayout.addWidget(tree)
                layout.addLayout(treeLayout)
            return layout

        metricsTreeWidget = self.createMetricsTree()
        layout.addWidget(metricsTreeWidget)
        numerical = QTreeWidgetItem(metricsTreeWidget, ['Numerical inputs'])
        self.categoricalRoot = QTreeWidgetItem(
            metricsTreeWidget, ['Categorical inputs']
        )

        self.populateMeasurementCatalog(numerical, isZstack, isSegm3D)
        self.categoricalRoot.setHidden(self.categoricalRoot.childCount() == 0)
        metricsTreeWidget.expandToDepth(0)
        if getattr(self, 'includeOtherMeasurements', False):
            self.otherMeasurementColumns = {}
            mainTree = metricsTreeWidget
            otherTree = self.createMetricsTree()
            for index, label in enumerate((
                    'Numerical other-cell inputs',
                    'Categorical other-cell inputs'
                )):
                sourceRoot = mainTree.topLevelItem(index)
                otherRoot = QTreeWidgetItem(otherTree, [label])
                self.copyOtherMeasurementGroups(sourceRoot, otherRoot)
            otherTree.expandToDepth(0)
            otherTreeLayout = QVBoxLayout()
            otherTreeLayout.addWidget(QLabel('Other-cell measurements'))
            otherTreeLayout.addWidget(otherTree)
            layout.addLayout(otherTreeLayout)
            self.otherMeasurementsTreeWidget = otherTree
        return layout

    def copyOtherMeasurementGroups(self, sourceParent, otherParent):
        for row in range(sourceParent.childCount()):
            sourceItem = sourceParent.child(row)
            if getattr(sourceItem, 'isCol', False):
                sourceName = sourceItem.variableName
                otherName = f'{sourceName}_other'
                item = QTreeWidgetItem(otherParent, [otherName])
                item.isCol = True
                item.setData(
                    0, widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,
                    otherName
                )
                item.variableName = otherName
                self.variableNames.append(otherName)
                self.otherMeasurementColumns[otherName] = sourceName
            else:
                group = QTreeWidgetItem(otherParent, [sourceItem.text(0)])
                self.copyOtherMeasurementGroups(sourceItem, group)

    @staticmethod
    def isNumericalInput(dtype):
        return (
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
        )

    def populateDataframeTree(self, numerical, categorical, df, index):
        nodes = {
            ('Numerical inputs',): numerical,
            ('Categorical inputs',): categorical,
        }
        for path, cols in measurements.measurement_column_groups(df, self.allChNames):
            for depth in range(2, len(path)+1):
                prefix = path[:depth]
                if prefix not in nodes:
                    nodes[prefix] = QTreeWidgetItem(
                        nodes[prefix[:-1]], [prefix[-1]]
                    )
            self.addTreeItems(nodes[path], cols, isCol=True, index=index)

    def createMetricsTree(self):
        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(fonts.font)
        self.metricsTreeWidget = metricsTreeWidget
        self.metricsTreeWidgets.append(metricsTreeWidget)
        metricsTreeWidget.itemDoubleClicked.connect(self.addColname)
        return metricsTreeWidget

    def populateMeasurementCatalog(self, root, isZstack, isSegm3D):
        """Fill the measurement catalog, keeping its groups.

        With an ``acdc_df`` and ``limitToLoadedColumns=True`` (measurements
        viewer) only catalog entries that are columns of the loaded table
        are listed, so the channel-less "All channels measurements" groups
        stay empty. With ``limitToLoadedColumns=False`` (main GUI and
        compute utility, which define measurements computed later) the
        full catalog is listed, including channel-less entries and
        measurements not saved yet. In both cases the table's remaining
        columns are added under "Other" and non-numerical columns go to
        "Categorical inputs"; the table also provides the test rows.
        """
        for chName in self.allChNames:
            channelTreeItem = QTreeWidgetItem(root)
            channelTreeItem.setText(0, f'{chName} measurements')

            metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )
            custom_metrics_desc = measurements.custom_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )

            foregrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            foregrMetricsTreeItem.setText(0, 'Cell measurements')
            channelTreeItem.addChild(foregrMetricsTreeItem)

            bkgrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            bkgrMetricsTreeItem.setText(0, 'Background values')
            channelTreeItem.addChild(bkgrMetricsTreeItem)

            if custom_metrics_desc:
                customMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                customMetricsTreeItem.setText(0, 'Custom measurements')
                channelTreeItem.addChild(customMetricsTreeItem)

            self.addTreeItems(
                foregrMetricsTreeItem, metrics_desc.keys(), isCol=True
            )
            self.addTreeItems(
                bkgrMetricsTreeItem, bkgr_val_desc.keys(), isCol=True
            )

            if custom_metrics_desc:
                self.addTreeItems(
                    customMetricsTreeItem, custom_metrics_desc.keys(),
                    isCol=True
                )

        self.addChannelLessItems(isZstack, isSegm3D=isSegm3D, root=root)

        sizeMetricsTreeItem = QTreeWidgetItem(root)
        sizeMetricsTreeItem.setText(0, 'Size measurements')

        size_metrics_desc = measurements.get_size_metrics_desc(
            isSegm3D, True
        )
        self.addTreeItems(
            sizeMetricsTreeItem, size_metrics_desc.keys(), isCol=True
        )

        propMetricsTreeItem = QTreeWidgetItem(root)
        propMetricsTreeItem.setText(0, 'Region properties')

        props_names = measurements.get_props_names()
        self.addTreeItems(
            propMetricsTreeItem, props_names, isCol=True
        )
        if self.acdcDf is not None:
            grouped = set(self.variableNames)
            for parent, numerical in (
                    (root, True), (self.categoricalRoot, False)
                ):
                remaining = [
                    col for col in self.acdcDf
                    if col not in grouped
                    and self.isNumericalInput(self.acdcDf[col].dtype) == numerical
                ]
                if remaining:
                    other = QTreeWidgetItem(parent, ['Other'])
                    self.addTreeItems(other, remaining, isCol=True)

    def addChannelLessItems(self, isZstack, isSegm3D=False, root=None):
        allChannelsTreeItem = QTreeWidgetItem(root)
        allChannelsTreeItem.setText(0, f'All channels measurements')
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, '', isSegm3D=isSegm3D
        )
        custom_metrics_desc = measurements.custom_metrics_desc(
            isZstack, '', isSegm3D=isSegm3D
        )
        # With an empty channel name the catalog also returns every saved
        # per-channel equation (e.g. `GFP_ratio`), which is already listed
        # under its channel. Channel-less names are `_<name>` (inserted
        # after each channel name), so keep only those.
        metrics_desc, bkgr_val_desc, custom_metrics_desc = (
            [name for name in desc if name.startswith('_')]
            for desc in (metrics_desc, bkgr_val_desc, custom_metrics_desc)
        )

        foregrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        foregrMetricsTreeItem.setText(0, 'Cell signal measurements')
        allChannelsTreeItem.addChild(foregrMetricsTreeItem)

        bkgrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        bkgrMetricsTreeItem.setText(0, 'Background values')
        allChannelsTreeItem.addChild(bkgrMetricsTreeItem)

        if custom_metrics_desc:
            customMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
            customMetricsTreeItem.setText(0, 'Custom measurements')
            allChannelsTreeItem.addChild(customMetricsTreeItem)

        self.addTreeItems(
            foregrMetricsTreeItem, metrics_desc, isCol=True,
            isChannelLess=True
        )
        self.addTreeItems(
            bkgrMetricsTreeItem, bkgr_val_desc, isCol=True,
            isChannelLess=True
        )

        if custom_metrics_desc:
            self.addTreeItems(
                customMetricsTreeItem, custom_metrics_desc,
                isCol=True, isChannelLess=True
            )

    def insertEquationText(self, text, checked=False):
        self.formulaEditWidget.insert(text)

    def clearEquation(self):
        self.formulaEditWidget.setText('')

    def initAttributes(self):
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        self.formulaEditWidget.backspace()

    def addTreeItems(
            self, parentItem, itemsText, isCol=False, isChannelLess=False,
            index=None
        ):
        for text in itemsText:
            parent = parentItem
            if (
                isCol and self.limitToLoadedColumns
                and self.acdcDfs is None and self.acdcDf is not None
                and text not in self.acdcDf
            ):
                continue
            if (
                self.acdcDfs is None and self.acdcDf is not None
                and text in self.acdcDf
                and not self.isNumericalInput(self.acdcDf[text].dtype)
                and parentItem.parent() is not self.categoricalRoot
            ):
                parent = self.categoricalRoot
            _item = QTreeWidgetItem(parent)
            _item.setText(0, text)
            if isCol:
                _item.isCol = True
                _item.setData(
                    0, widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE, text
                )
                variable = text if index is None else f'{text}_table{index+1}'
                _item.variableName = variable
                # A name listed in more than one group is one variable;
                # a channel item always wins over a channel-less one.
                if variable not in self._listedVariables:
                    self._listedVariables.add(variable)
                    self.variableNames.append(variable)
                    if isChannelLess:
                        self.channelLessVariables.add(variable)
                elif not isChannelLess:
                    self.channelLessVariables.discard(variable)
            _item.isChannelLess = isChannelLess


    def addColname(self, item, column):
        if not hasattr(item, 'isCol'):
            return

        colName = item.variableName
        if not colName.isidentifier() or keyword.iskeyword(colName):
            colName = f'`{colName}`'
        self.insertEquationText(colName)

    def _debug(self):
        print(self.formulaEditWidget.text(), self.equationColNames)

    @timepoint_debug_trace
    def equationChanged(self, *args):
        self.okButton.setDisabled(True)
        self.statusLabel.clear()
        self.testOutputDisplay.clear()
        self.testOutputDisplay.setStyleSheet('')
        self.equationColNames = self.formulaEditWidget.referencedVariableNames()
        self.channelLessColnames = [
            name for name in self.equationColNames
            if name in self.channelLessVariables
        ]

    def getTestData(self, preview=False):
        if self.acdcDfs is not None:
            return pd.concat([
                (df.head(10) if preview else df).add_suffix(f'_table{i+1}')
                for i, df in enumerate(self.acdcDfs.values())
            ], axis=1)
        names = self.formulaEditWidget.referencedVariableNames()
        if self.acdcDf is None:
            sample = pd.DataFrame(index=[0])
        elif self.allowCellAggregations:
            # Per-cell functions need the whole table (all rows/columns)
            sample = (
                self.acdcDf.head(10) if preview else self.acdcDf
            ).copy()
        else:
            # Row-wise equations: copy only the referenced columns of a
            # limited number of rows (the test shows only 10 results and
            # the equation is evaluated again on the full table at save)
            nrows = 10 if preview else self.TEST_DATA_MAX_ROWS
            cols = [name for name in names if name in self.acdcDf.columns]
            sample = self.acdcDf[cols].head(nrows).copy()
        if not self.limitToLoadedColumns:
            # Main GUI / compute utility: yes/no columns as at save
            # (`cli.ComputeMeasurementsKernel._df_eval_equation`)
            sample = load.bool_cols_as_plain_bool(sample)
        for name in names:
            if name not in sample and name not in sample.index.names:
                sample[name] = np.random.uniform(1, 5, len(sample))
        if self.allowCellAggregations:
            return sample
        return sample[names]

    def validateEquation(self, formula):
        try:
            if not formula.strip():
                return False, 'Equation cannot be empty'
            if not self.formulaEditWidget.referencedVariableNames():
                return False, 'No measurement inputs used'
            result = self.evaluateEquation(
                self.getTestData(preview=True), formula
            )
            if not isinstance(result, pd.Series):
                return False, 'The equation must return one value per row'
        except Exception as err:
            return False, str(err)
        return True, formula

    @timepoint_debug_trace
    def evaluateEquation(
            self, df, formula, *, enforce_dtype_output=None, alternatives=None
        ):
        """Use downstream pandas/per-cell execution and optionally check dtype."""
        from cellacdc import timepoint_predictors
        if not self.allowCellAggregations:
            if re.search(r'\b(?:any|all)\s*\(', formula):
                raise ValueError(
                    'any(...) and all(...) are available only in the '
                    'Measurements viewer.'
                )
            result = df.eval(formula)
            if enforce_dtype_output is not None:
                result = timepoint_predictors.CellEquation.enforce_dtype(
                    result, enforce_dtype_output, len(df)
                )
            return result

        return self.evaluateCellEquation(
            df, formula, SizeT=getattr(self, 'SizeT', None),
            enforce_dtype_output=enforce_dtype_output, alternatives=alternatives
        )

    @staticmethod
    def isRowWiseFormula(formula):
        """See ``timepoint_predictors.CellEquation.is_row_wise``."""
        from cellacdc import timepoint_predictors
        return timepoint_predictors.CellEquation.is_row_wise(formula)

    @staticmethod
    def evaluateCellEquation(
            df, formula, *, SizeT=None, enforce_dtype_output=None,
            alternatives=None
        ):
        """Evaluate viewer equations without requiring an equation editor.

        Formulas with per-cell functions (``CELL_FUNCTIONS``) or other-cell
        ``alternatives`` are evaluated cell by cell (rows sorted by
        frame_i). Purely row-wise formulas give the same result evaluated
        once on the whole table, so they take a vectorised fast path.
        """
        from cellacdc import timepoint_predictors
        variables = {'SizeT': SizeT} if SizeT is not None else None
        flat = timepoint_predictors.CellTable.flatten(df)
        if 'Cell_ID' not in flat or flat.empty:
            return timepoint_predictors.CellEquation.evaluate(
                flat, formula, variables,
                enforce_dtype_output=enforce_dtype_output
            )
        if (
            not alternatives and not flat['Cell_ID'].isna().any()
            and _CombineMetricsDialog.isRowWiseFormula(formula)
        ):
            try:
                values = timepoint_predictors.CellEquation.evaluate(
                    flat, formula, variables,
                    enforce_dtype_output=enforce_dtype_output
                )
            except Exception as err:
                raise ValueError(
                    f'Evaluating the equation failed: {err}'
                ) from err
            # Same conversion as the per-cell path below
            result = pd.Series(
                np.broadcast_to(np.asarray(values), len(flat)).copy(),
                index=df.index
            )
            return result
        results = []
        for ID, cell in flat.groupby('Cell_ID', sort=False):
            if 'frame_i' in cell:
                cell = cell.sort_values('frame_i')
            try:
                values = (
                    timepoint_predictors.OtherCellMeasurements
                    .evaluate_with_alternatives(
                        cell.reset_index(drop=True), formula, alternatives,
                        variables, enforce_dtype_output=enforce_dtype_output
                    )
                )
                results.append(pd.Series(
                    np.broadcast_to(np.asarray(values), len(cell)),
                    index=cell.index
                ))
            except Exception as err:
                raise ValueError(
                    f'Evaluating the equation failed for Cell_ID {ID}: {err}'
                ) from err
        result = pd.concat(results).reindex(flat.index)
        result.index = df.index
        return result

    def test_cb(self):
        self.equationChanged()
        success, message = self.formulaEditWidget.checkValidity()
        if not success:
            self.statusLabel.setText(message)
            self.testOutputDisplay.setPlainText(message)
            self.testOutputDisplay.setStyleSheet('border: 2px solid red')
            return
        try:
            df = self.getTestData()
            result = self.evaluateEquation(
                df, self.formulaEditWidget.text()
            )
        except Exception as err:
            traceback.print_exc()
            self.statusLabel.setText(str(err))
            self.testOutputDisplay.setPlainText(str(err))
            self.testOutputDisplay.setStyleSheet('border: 2px solid red')
            return
        preview = df[self.equationColNames].head(10).copy()
        preview[self.newColNameLineEdit.text() or 'Result'] = result.head(10)
        self.testOutputDisplay.setHtml(preview.to_html())
        self.testOutputDisplay.setStyleSheet('border: 2px solid green')
        source = 'loaded tables' if self.acdcDfs is not None else 'random inputs'
        if self.acdcDf is not None:
            source = 'sampled table rows (random inputs for new measurements)'
        self.statusLabel.setText(f'Equation test with {source} was successful.')
        self.okButton.setEnabled(True)
        if hasattr(self, 'logger'):
            self.logger.info(self.statusLabel.text())

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        msg.critical(
            self, 'Empty new measurement name',
            html_utils.paragraph('"New measurement name" field cannot be empty!')
        )

    def showHelp(self):
        txt = measurements.get_combine_metrics_help_txt()
        msg = widgets.myMessageBox(
            showCentered=False, wrapText=False,
            scrollableText=True, enlargeWidthFactor=1.7
        )
        msg.addShowInFileManagerButton(
            measurements.acdc_metrics_path, txt='Show saved file...'
        )
        msg.information(self, 'Combine measurements help', txt)


class CombineMetricsEquationDialog(_CombineMetricsDialog):
    sigOk = Signal(object)

    def __init__(
            self, allChNames, isZstack, isSegm3D, parent=None, debug=False,
            closeOnOk=True, acdc_df=None, *, saveToConfig=True,
            allowCellAggregations=False, limitToLoadedColumns=False
        ):
        self.saveToConfig = saveToConfig
        super().__init__(
            allChNames, isZstack, isSegm3D, parent=parent, debug=debug,
            closeOnOk=closeOnOk, acdc_df=acdc_df,
            allowCellAggregations=allowCellAggregations,
            limitToLoadedColumns=limitToLoadedColumns
        )

    def getEquationsDict(self):
        equation = self.equationDisplay.toPlainText()
        newColName = self.newColNameLineEdit.text()
        if not self.channelLessColnames:
            chNamesInTerms = set()
            for term in self.equationColNames:
                for chName in self.allChNames:
                    if term.startswith(f'{chName}_'):
                        chNamesInTerms.add(chName)
            if len(chNamesInTerms) == 1:
                # Equation uses metrics from a single channel --> append channel name
                chName = chNamesInTerms.pop()
                chColName = f'{chName}_{newColName}'
                isMixedChannels = False
                return {chColName:equation}, isMixedChannels
            else:
                # Equation doesn't use all channels metrics nor is single channel
                isMixedChannels = True
                return {newColName:equation}, isMixedChannels

        isMixedChannels = False
        equations = {}
        for chName in self.allChNames:
            replacements = {
                colName: f'{chName}{colName}'
                for colName in self.channelLessColnames
            }
            equations[f'{chName}_{newColName}'] = (
                self.formulaEditWidget.replaceVariableNames(replacements)
            )
        return equations, isMixedChannels

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        if not self.okButton.isEnabled():
            self.test_cb()
            return
        
        if self.saveToConfig:
            self.saveEquationsConfig()
        self.cancel = False
        self.sigOk.emit(self)

        if self.closeOnOk:
            self.close()

    def saveEquationsConfig(self):
        # Save equation to "<user_profile_path>/acdc-metrics/combine_metrics.ini" file
        config = measurements.read_saved_user_combine_config()

        equationsDict, isMixedChannels = self.getEquationsDict()
        for newColName, equation in equationsDict.items():
            config = measurements.add_user_combine_metrics(
                config, equation, newColName, isMixedChannels
            )

        isChannelLess = len(self.channelLessColnames) > 0
        if isChannelLess:
            channelLess_equation = self.equationDisplay.toPlainText()
            equation_name = self.newColNameLineEdit.text()
            config = measurements.add_channelLess_combine_metrics(
                config, channelLess_equation, equation_name,
                self.channelLessColnames
            )

        measurements.save_common_combine_metrics(config)


class CombineFeaturesCalculator(dialogs_base.QBaseDialog):
    sigOk = Signal(object)
    
    def __init__(
            self, features_groups: dict, 
            group_name_to_col_mapper: dict=None,
            title='Combine features calculator',
            parent=None
        ):
        super().__init__(parent)
        
        self.cancel = True
        
        self.setWindowTitle(title)
        self.initAttributes()
        
        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()
        
        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(fonts.font)
        self.metricsTreeWidget = metricsTreeWidget
        
        for groupName, features in features_groups.items():
            topLevelTreeWidgetItem = QTreeWidgetItem(metricsTreeWidget)
            topLevelTreeWidgetItem.setText(0, groupName)
            metricsTreeWidget.addTopLevelItem(topLevelTreeWidgetItem)
            self.addTreeItems(
                topLevelTreeWidgetItem, features, isCol=True, 
                name_to_col_mapper=group_name_to_col_mapper.get(groupName)
            )
            
        operatorsLayout = self.createOperatorsLayout()
        newFeatureNameLayout = self.createNewFeatureNameLayout()
        equationDisplayLayout = self.createEquationDisplayLayout()
        
        equationLayout.addLayout(newFeatureNameLayout)
        equationLayout.addWidget(QLabel(' = '))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0,1)
        equationLayout.setStretch(1,0)
        equationLayout.setStretch(2,2)
        
        testOutputLayout = self.createTestOutputLayout()
        buttonsLayout = self.createButtonsOutputLayout()
        
        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            Before clicking the `Ok` button, check that the equation returns 
            the expected result by clicking the `Test output` button. 
        """)
        
        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(QLabel('Available measurements:'))
        mainLayout.addWidget(metricsTreeWidget)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(testOutputLayout)

        
        metricsTreeWidget.itemDoubleClicked.connect(self.addFeatureName)
        self.setLayout(mainLayout)
        self.setFont(fonts.font)

        self.setStyleSheet(widgets_style.TREEWIDGET_STYLESHEET)
    
    def setExpandedAll(self, expanded):
        if expanded:
            self.expandAll()
        else:
            for i in range(self.metricsTreeWidget.topLevelItemCount()):
                topLevelItem = self.metricsTreeWidget.topLevelItem(i)
                topLevelItem.setExpanded(False)
    
    def expandAll(self):
        for i in range(self.metricsTreeWidget.topLevelItemCount()):
            topLevelItem = self.metricsTreeWidget.topLevelItem(i)
            topLevelItem.setExpanded(True)
    
    def addTreeItems(
            self, parentItem, itemsText, isCol=False, name_to_col_mapper=None
        ):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            _item.variable_name = text
            if name_to_col_mapper is None:
                continue
            
            col_name = name_to_col_mapper.get(text, None)
            if col_name is None:
                continue
            
            _item.variable_name = col_name
            
    
    def addFeatureName(self, item, column):
        if not hasattr(item, 'isCol'):
            return

        colName = item.variable_name
        text = f'{self.equationDisplay.toPlainText()}{colName}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
    
    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()
    
    def createButtonsOutputLayout(self):
        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.infoPushButton('  Help...')
        testButton = widgets.calcPushButton('Test output')
        okButton = widgets.okPushButton(' Ok ')
        okButton.setDisabled(True)
        self.okButton = okButton
        okButton.setToolTip('Click "Test output" to check the equation and enable this button')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)
        
        helpButton.clicked.connect(self.showHelp)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)
        
        return buttonsLayout

    def ok_cb(self):
        if not self.newFeatureNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        
        self.equation = self.equationDisplay.toPlainText()
        self.newFeatureName = self.newFeatureNameLineEdit.text()
        self.cancel = False
        self.close()
        self.sigOk.emit(self)
    
    def test_cb(self):
        # Evaluate equation with random inputs
        equation = self.equationDisplay.toPlainText()
        random_data = np.random.rand(1, len(self.equationColNames))*5
        df = pd.DataFrame(
            data=random_data,
            columns=self.equationColNames
        ).round(5)
        newColName = self.newFeatureNameLineEdit.text()
        try:
            df[newColName] = df.eval(equation)
        except Exception as e:
            traceback.print_exc()
            self.testOutputDisplay.setHtml(html_utils.paragraph(e))
            self.testOutputDisplay.setStyleSheet("border: 2px solid red")
            return

        self.testOutputDisplay.setStyleSheet("border: 2px solid green")
        self.okButton.setDisabled(False)

        result = df.round(5).iloc[0][newColName]

        # Substitute numbers into equation
        inputs = df.iloc[0]
        equation_numbers = equation
        for c, col in enumerate(self.equationColNames):
            equation_numbers = equation_numbers.replace(col, str(inputs[c]))

        # Format output into html text
        cols = self.equationColNames
        inputs_txt = [f'{col} = {input}' for col, input in zip(cols, inputs)]
        list_html = html_utils.to_list(inputs_txt)
        text = html_utils.paragraph(f"""
            By substituting the following random inputs:
            {list_html}
            we get the equation:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {equation_numbers}</code><br><br>
            that <b>equals to</b>:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {result}</code>
        """)
        self.testOutputDisplay.setHtml(text)
    
    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(
            self, 'Empty new measurement name', txt
        )
    
    def showHelp(self):
        pass
    
    def createTestOutputLayout(self):
        testOutputLayout = QVBoxLayout()
        testOutputLayout.addWidget(QLabel('Result of test with random inputs:'))
        testOutputDisplay = QTextEdit()
        testOutputDisplay.setReadOnly(True)
        self.testOutputDisplay = testOutputDisplay
        testOutputLayout.addWidget(testOutputDisplay)
        testOutputLayout.setStretch(0,0)
        testOutputLayout.setStretch(1,1)
        
        return testOutputLayout
    
    def createEquationDisplayLayout(self):
        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel('Equation:'))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0,0)
        equationDisplayLayout.setStretch(1,1)
        return equationDisplayLayout
    
    def createNewFeatureNameLayout(self):
        newFeatureNameLayout = QVBoxLayout()
        newFeatureNameLineEdit = widgets.alphaNumericLineEdit()
        newFeatureNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newFeatureNameLineEdit = newFeatureNameLineEdit
        newFeatureNameLayout.addStretch(1)
        newFeatureNameLayout.addWidget(QLabel('New measurement name:'))
        newFeatureNameLayout.addWidget(newFeatureNameLineEdit)
        newFeatureNameLayout.addStretch(1)
        return newFeatureNameLayout
    
    def createOperatorsLayout(self):
        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ('add', '+'),
            ('subtract', '-'),
            ('multiply', '*'),
            ('divide', '/'),
            ('open_bracket', '('),
            ('close_bracket', ')'),
            ('square', '**2'),
            ('pow', '**'),
            ('ln', 'log('),
            ('log10', 'log10('),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f':{name}.svg'))
            button.setIconSize(QSize(iconSize,iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(':clear.svg'))
        clearButton.setIconSize(QSize(iconSize,iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(':backspace.svg'))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize,iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)
        
        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)
        
        return operatorsLayout

    def addOperator(self):
        button = self.sender()
        text = f'{self.equationDisplay.toPlainText()}{button.text}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()
    
    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []
    
    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[:-self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1]:]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)


class FormulaEditWidget(QWidget):
    sigFormulaChanged = Signal(str, bool)  # formula_str, is_valid

    def __init__(
            self, variable_names=None, parent=None, *,
            validator=None, allow_empty=True
        ):
        super().__init__(parent)
        self._variable_names = variable_names or []
        self._validator = validator
        self._allow_empty = allow_empty
        self._live_validation = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText('e.g. img1 + img2 * 0.5')
        layout.addWidget(self._edit)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet('font-size: 11px;')
        layout.addWidget(self._status_label)

        self._edit.textChanged.connect(self._onTextChanged)
        self._clearStatus()
        
    def setVariableNames(self, variable_names):
        """Allows setting the variables.

        Parameters
        ----------
        variable_names : list
            list of variable names (strings)
        """
        
        self._variable_names = variable_names
        self._onTextChanged(self._edit.text())

    def text(self):
        """Returns the current formula text."""
        return self._edit.text()

    def setText(self, text):
        """Sets the formula text."""
        self._edit.setText(text)

    def setLiveValidation(self, enabled):
        """Enable or disable running the equation validator while typing."""
        self._live_validation = bool(enabled)
        if not self._live_validation:
            self._clearStatus()

    def toPlainText(self):
        return self.text()

    def setPlainText(self, text):
        self.setText(text)

    def insert(self, text):
        self._edit.insert(text)

    def backspace(self):
        self._edit.backspace()

    def referencedVariableNames(self):
        """Return known input names, excluding names inside string literals."""
        names = {
            name for name, _, _ in measurements.equation_variable_spans(self.text())
        }
        return [name for name in self._variable_names if name in names]

    def replaceVariableNames(self, replacements):
        return measurements.replace_equation_variables(self.text(), replacements)

    def _clearStatus(self):
        self._status_label.setText('')
        self._status_label.setStyleSheet('font-size: 11px;')

    @timepoint_debug_trace
    def _onTextChanged(self, text):
        if not self._live_validation:
            self._clearStatus()
            self.sigFormulaChanged.emit(text, False)
            return
        if not text.strip():
            self._clearStatus()

        success, reconstructed_str = self.checkValidity(self._variable_names)

        if success:
            self._status_label.setText(f'→ {reconstructed_str}')
            self._status_label.setStyleSheet(
                'font-size: 11px; color: green;'
            )
        else:
            self._status_label.setText(reconstructed_str)
            self._status_label.setStyleSheet(
                'font-size: 11px; color: red;'
            )

        self.sigFormulaChanged.emit(text, success)

    def checkValidity(self, variable_names=None):
        """Use a custom evaluator, or SymPy for numerical image formulas."""
        if not self.text().strip() and not self._allow_empty:
            return False, 'Equation cannot be empty'
        if self._validator is not None:
            return self._validator(self.text())
        return self.checkSymbolicValidity(variable_names)

    def checkSymbolicValidity(self, variable_names=None):
        if variable_names is None:
            variable_names = self._variable_names
        formula_str = self._edit.text()
        arrays = {name: 1 for name in variable_names}
        success = False
        reconstructed_str = 'ERROR'
        if not formula_str.strip():
            if self._allow_empty:
                return True, 'First channel is returned/applied'
            return False, 'Equation cannot be empty'
        try:
            symbols = {name: sp.Symbol(name) for name in arrays}
            expr = sp.sympify(formula_str, locals=symbols)
            missing = {str(s) for s in expr.free_symbols} - arrays.keys()
            if missing:
                reconstructed_str = f'Missing variables: {missing}'
                return False, reconstructed_str

            # filter out expressions that have no variables
            if not any(s.is_Symbol for s in expr.free_symbols):
                reconstructed_str = 'No variables used'
                return False, reconstructed_str
            
            reconstructed_str = str(expr)
            success = True
        except Exception as e:
            if 'syntax' in str(e):
                reconstructed_str = f'Syntax error'
            else:
                reconstructed_str = str(e)
            success = False
        return success, reconstructed_str
