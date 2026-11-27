"""C14: docstrings, comments and user-visible strings of cellacdc/ that
still name old modules after the C12 rewrite (rewrite.py lists them as
`mentions` and never rewrites them).

Rules:
- references to classes and functions use the canonical alias of their
  new module, as the code does (`widgets_message_box.myMessageBox`,
  `dialogs_set_measurements.SetMeasurementsDialog`); the aliases of the
  shared GUI mixin modules keep the old names (`gui_navigation`,
  `gui_overlay`, ...), so mentions through them stay as they are unless
  the name moved to another module (`gui_window_core.GUICoreElements`);
- the main GUI window and the viewer window are named by their classes
  (`guiWin`, `MeasurementsViewerWin`), never `gui.guiWin` or
  `measurements_viewer.MeasurementsViewerWin` (old paths that warn);
- user-visible texts name the module in words ("the data prep module",
  CLAUDE.md);
- numpydoc types use the full dotted path of the new module;
- plugin API names are referenced at their defining module in cellacdc's
  own docs (myutils.<name> keeps working for plugins);
- provenance notes of the generated module docstrings ("from apps.py",
  "was myutils_models.py") and the compat code naming old modules stay.

Not changed: logger names ('dataPrep', 'dataStruct-*': log file names) and
data values ('dataPrep' background), commented-out code without a new home
(apps.CellConstantsDialog, myutils._download_trackastra_models in
_run.py: neither exists), the dev scripts in _profile/ and debugutils.py's
Path("cellacdc/gui.py") (code, not a mention). Module aliases that the C12
rewrite kept with old names (`as dataPrep`, `as segm`, `as _main`, ...)
are code: comments name the canonical alias of the module.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

GUIWIN = ('`gui.guiWin', '`guiWin')
VIEWERWIN = ('`measurements_viewer.MeasurementsViewerWin`',
             '`MeasurementsViewerWin`')

# (file, [(old, new, count)], items logged for behaviour-relevant edits)
EDITS = [
    # --- user-visible strings
    ('cellacdc/data_prep/main_window.py', [
        ("'_segmInfo.csv file not found. Launching dataPrep.py...'",
         "'_segmInfo.csv file not found. Launching the data prep module...'",
         1),
    ], ['cellacdc/data_prep/main_window.py::launchDataPrepSelectZPlane']),
    # as 'Closing data structure logger...', 'Closing segmentation module
    # logger...' of the other modules
    ('cellacdc/data_prep/main_window.py', [
        ("self.logger.info('Closing dataPrep logger...')",
         "self.logger.info('Closing data prep module logger...')", 1),
    ], ['cellacdc/data_prep/main_window.py::dataPrepWin.closeEvent']),
    ('cellacdc/gui/mixins/overlay.py', [
        ("'Run the script \"dataPrep.py\" to create the following file:\\n\\n'",
         "'Run the data prep module to create the following file:\\n\\n'",
         1),
    ], ['cellacdc/gui/mixins/overlay.py::'
        'OverlayChannelsGuiElements.loadNonAlignedFluoChannel']),
    ('cellacdc/gui/mixins/overlay.py', [
        ('(`gui.guiWin`) and the measurements viewer\n'
         '(`measurements_viewer.MeasurementsViewerWin`).',
         '(`guiWin`) and the measurements viewer\n'
         '(`MeasurementsViewerWin`).', 1),
        ('`widgets.OverlayToolbar`', '`widgets_toolbars.OverlayToolbar`', 1),
        ('(`gui_viewer_tools.DefaultSettingsGuiElements.restoreDefaultSettings`).',
         '(`gui_window_core.DefaultSettingsGuiElements.restoreDefaultSettings`).',
         1),
    ], []),
    ('cellacdc/service/worker.py', [
        ("            'client must send the data instead (see '\n"
         "            'myutils.resolve_segm_model_init_data).'\n",
         "            'client must send the data instead (see '\n"
         "            'resolve_segm_model_init_data in myutils/segm_models.py).'\n",
         1),
    ], ['cellacdc/service/worker.py::instantiate_model']),
    # --- docstrings and comments
    ('cellacdc/_types.py', [
        ('    cellacdc.widgets.VectorLineEdit widget in the automatic GUI.',
         '    VectorLineEdit widget (shared/widgets/line_edits.py) in the '
         'automatic GUI.', 1),
    ], []),
    ('cellacdc/background_values.py', [
        ('objects..." (`utils/computeBackgroundValues.py`, worker\n'
         '`workers.ComputeBackgroundValuesWorker`).',
         'objects..." (`utils/computeBackgroundValues.py`, worker\n'
         '`utils_computeBackgroundValues.ComputeBackgroundValuesWorker`).', 1),
        ('(`apps_measurements_viewer.ChannelPlaceholderMappingDialog`).',
         '(`viewer_dialogs_premade_setup.ChannelPlaceholderMappingDialog`).',
         1),
        ('# `workers.ComputeBackgroundValuesWorker._correctAcdcOutput` and the',
         "# `utils_computeBackgroundValues.ComputeBackgroundValuesWorker`'s\n"
         '# `_correctAcdcOutput` and the', 1),
    ], []),
    ('cellacdc/cli.py', [
        ('`apps.SetMeasurementsDialog`',
         '`dialogs_set_measurements.SetMeasurementsDialog`', 1),
        ('`apps.SetMeasurementsDialog.ok_cb`',
         '`dialogs_set_measurements.SetMeasurementsDialog.ok_cb`', 1),
    ], []),
    ('cellacdc/gui/main_window/layout.py', [
        ("        # gui_overlay.addAlphaScrollbar) that cannot see gui.py's _font",
         "        # gui_overlay.addAlphaScrollbar) that cannot read the _font of\n"
         "        # gui/main_window/constants.py", 1),
    ], []),
    ('cellacdc/gui/main_window/menus_actions.py', [
        ('# self.toggleTerminalButton = widgets.ToggleTerminalButton()',
         '# self.toggleTerminalButton = '
         'widgets_push_buttons.ToggleTerminalButton()', 1),
    ], []),
    ('cellacdc/gui/main_window/points_layers.py', [
        ('# apps.AddPointsLayerDialog.ok_cb()',
         '# gui_dialogs_points_layers.AddPointsLayerDialog.ok_cb()', 1),
    ], []),
    ('cellacdc/gui/main_window/rois.py', [
        ('(`widgets.PromptBoxROI`)',
         '(`gui_widgets_points_layer_widgets.PromptBoxROI`)', 1),
    ], []),
    ('cellacdc/gui/main_window/window_setup.py', [
        ('# `widgets.PromptBoxROI` items',
         '# `gui_widgets_points_layer_widgets.PromptBoxROI` items', 1),
    ], []),
    ('cellacdc/gui/mixins/annotations.py', [
        ('It is shared by the main GUI (`gui.guiWin`) and the measurements '
         'viewer\n(`measurements_viewer.MeasurementsViewerWin`).',
         'It is shared by the main GUI (`guiWin`) and the measurements '
         'viewer\n(`MeasurementsViewerWin`).', 1),
        ('(`widgets.myHistogramLUTitem`)',
         '(`widgets_lut_items.myHistogramLUTitem`)', 1),
        ('(`widgets.labelsGradientWidget`)',
         '(`widgets_lut_items.labelsGradientWidget`)', 1),
        ('ax : cellacdc.widgets.MainPlotItem',
         'ax : cellacdc.shared.widgets.plot_items.MainPlotItem', 1),
    ], []),
    ('cellacdc/gui/mixins/custom_annotations.py', [
        ('It is shared by the main GUI (`gui.guiWin`) and the measurements '
         'viewer \n(`measurements_viewer.MeasurementsViewerWin`).',
         'It is shared by the main GUI (`guiWin`) and the measurements '
         'viewer \n(`MeasurementsViewerWin`).', 1),
        ('(`widgets.CustomAnnotToolButton`)',
         '(`gui_mixin_widgets.CustomAnnotToolButton`)', 1),
        ('`gui_core.GUICoreElements`', '`gui_window_core.GUICoreElements`', 1),
        ('`apps.customAnnotationDialog`',
         '`gui_mixin_dialogs.customAnnotationDialog`', 2),
    ], []),
    ('cellacdc/gui/mixins/dialogs.py', [
        ('See `apps_measurements_viewer._CustomAnnotationPredictorGroupbox`.',
         'See `viewer_dialogs_predictor_editors.'
         '_CustomAnnotationPredictorGroupbox`.', 1),
        ('            # Measurements viewer only; local import avoids a circular\n'
         '            # import (apps_measurements_viewer imports apps)\n',
         '            # Measurements viewer only; imported here: gui/mixins/ '
         'imports\n'
         '            # the viewer only lazily (see gui/mixins/__init__.py)\n',
         1),
    ], []),
    ('cellacdc/gui/mixins/navigation.py', [
        ('shared between the main GUI (`gui.guiWin`) and other windows (e.g., '
         '\n`measurements_viewer.MeasurementsViewerWin`).',
         'shared between the main GUI (`guiWin`) and other windows (e.g., '
         '\n`MeasurementsViewerWin`).', 1),
        ('(`widgets.ParentImageItem`)',
         '(`widgets_plot_items.ParentImageItem`)', 1),
        ('(`widgets.myHistogramLUTitem`)',
         '(`widgets_lut_items.myHistogramLUTitem`)', 1),
        ('(`widgets.baseHistogramLUTitem.colorMapMenuClicked`)',
         '(`widgets_lut_items.baseHistogramLUTitem.colorMapMenuClicked`)', 1),
    ], []),
    ('cellacdc/gui/mixins/viewer_tools.py', [
        ('"""Mixin classes with the \'Viewer\' mode tools of '
         '`cellacdc.gui.guiWin` \n'
         'shared with other windows (e.g.,\n'
         '`cellacdc.measurements_viewer.MeasurementsViewerWin`): zoom, ruler, '
         'scale bar and timestamp, color scheme, labels colormap, image \n'
         'properties, cell cycle table, objects count, search ID, hover info, '
         'keyboard \n'
         'shortcuts, default settings, slideshow, 3D viewer and region '
         'properties \n'
         'dock widget.\n'
         '"""',
         '"""Mixin classes with the \'Viewer\' mode tools of `guiWin` shared '
         'with\n'
         'other windows (e.g., `MeasurementsViewerWin`): zoom, ruler, scale '
         'bar\n'
         'and timestamp, color scheme, labels colormap, image properties,\n'
         'slideshow, 3D viewer and the bottom layout. The cell cycle table,\n'
         'objects count, search ID, hover info and region properties dock '
         'are in\n'
         'gui/mixins/object_info.py, the keyboard shortcuts and default '
         'settings\n'
         'in gui/mixins/window_core.py.\n'
         '"""', 1),
        ('(`widgets.ScrollArea`)', '(`widgets_basic_widgets.ScrollArea`)', 1),
        ('(`widgets.VerticalResizeHline`)',
         '(`widgets_basic_widgets.VerticalResizeHline`)', 1),
    ], []),
    ('cellacdc/gui/workers/loading_saving.py', [
        ('# see loadSelectedData function in gui.py',
         '# see loadSelectedData in gui/main_window/loading.py', 1),
    ], []),
    ('cellacdc/launcher/main_window.py', [
        ('# gui.custom_annot_path, ',
         '# gui_custom_annotations.custom_annot_path, ', 1),
        ('# logoDialog = apps.LogoDialog(',
         '# logoDialog = launcher_dialogs.LogoDialog(', 1),
    ], []),
    ('cellacdc/lineage_tree_integrity.py', [
        ('root cells by ``myutils.df_ctc_to_acdc_df``, e.g. Trackastra '
         'output).',
         'root cells by ``table_utils.df_ctc_to_acdc_df``, e.g. Trackastra '
         'output).', 1),
        ('(`workers.ConcatAcdcDfsWorker`)',
         '(`utils_concat.ConcatAcdcDfsWorker`)', 1),
    ], []),
    ('cellacdc/load.py', [
        ('(`apps._CombineMetricsDialog.getTestData`)',
         '(`dialogs_combine_measurements._CombineMetricsDialog.getTestData`)',
         1),
    ], []),
    ('cellacdc/measurements.py', [
        ('(`apps.SetMeasurementsDialog`)',
         '(`dialogs_set_measurements.SetMeasurementsDialog`)', 1),
    ], []),
    ('cellacdc/measurements_viewer/__init__.py', [
        ("The widget-free code (data loading and the window's data/logic "
         "methods,\n"
         "`measurements_viewer_utils.MeasurementsViewerUtils`, inherited by\n"
         "`MeasurementsViewerWin`) lives in `measurements_viewer_utils`; this\n"
         "docstring specifies both modules.",
         "The widget-free code (data loading and the window's data/logic "
         "methods,\n"
         "`viewer_utils.MeasurementsViewerUtils` and its bases, inherited by\n"
         "`MeasurementsViewerWin`) lives in `viewer_utils`, `position_data` "
         "and\n"
         "`annotation_data`; this docstring specifies the window and these "
         "modules.", 1),
        ('the main GUI (`gui.guiWin`) through the mixins in:',
         'the main GUI (`guiWin`) through the mixins in:', 1),
        ("- `gui_viewer_tools`: the other 'Viewer' mode tools (zoom, ruler, "
         "scale bar, \n"
         "  timestamp, colormaps, search ID, hover info, shortcuts, "
         "slideshow, \n"
         "  3D viewer, objects properties, cell cycle table and objects "
         "count). The\n",
         "- `gui_viewer_tools`: the other 'Viewer' mode tools (zoom, ruler, "
         "scale bar,\n"
         "  timestamp, colormaps, slideshow, 3D viewer), `gui_object_info` "
         "(search\n"
         "  ID, hover info, objects properties, cell cycle table and objects "
         "count)\n"
         "  and `gui_window_core` (shortcuts and default settings). The\n", 1),
        ('both `_main.mainWin.launchMeasurementsViewer`)',
         'both `launcher_window.mainWin.launchMeasurementsViewer`)', 1),
        ('(`gui.guiWin.setStatusBarLabel`)', '(`guiWin.setStatusBarLabel`)',
         1),
        ('`measurements_viewer_utils.loaded_data_status`',
         '`viewer_utils.loaded_data_status`', 1),
        ('`measurements_viewer_utils.path_tooltip_lines`',
         '`viewer_utils.path_tooltip_lines`', 1),
        ('`widgets.getPushButton`', '`widgets_push_buttons.getPushButton`',
         1),
        ('`apps.customAnnotationDialog`',
         '`gui_mixin_dialogs.customAnnotationDialog`', 1),
    ], []),
    ('cellacdc/measurements_viewer/dialogs/phase_translator_dialogs.py', [
        ('# of the post-processing dialog in apps.py',
         '# of the post-processing dialog (gui/dialogs/segmentation.py)', 1),
    ], []),
    ('cellacdc/measurements_viewer/main_window/custom_annotations.py', [
        GUIWIN + (1,)], []),
    ('cellacdc/measurements_viewer/main_window/data_io.py', [
        GUIWIN + (1,)], []),
    ('cellacdc/measurements_viewer/main_window/image_panel.py', [
        GUIWIN + (9,)], []),
    ('cellacdc/measurements_viewer/main_window/navigation.py', [
        GUIWIN + (2,)], []),
    ('cellacdc/measurements_viewer/main_window/timepoints.py', [
        ('#     msg = widgets.myMessageBox(wrapText=False)',
         '#     msg = widgets_message_box.myMessageBox(wrapText=False)', 1),
    ], []),
    ('cellacdc/measurements_viewer/main_window/ui_setup.py', [
        GUIWIN + (3,),
        ('# rows. The main GUI builds its own navigation layout (gui.py).',
         '# rows. The main GUI builds its own navigation layout\n'
         '        # (gui/main_window/layout.py).', 1),
    ], []),
    ('cellacdc/measurements_viewer/position_data.py', [
        GUIWIN + (6,)], []),
    ('cellacdc/measurements_viewer/viewer_utils.py', [
        VIEWERWIN + (1,),
        ('plots or widgets belong to `measurements_viewer`. The behavior of '
         'these\n'
         'functions is specified in the `measurements_viewer` module '
         'docstring.',
         'plots or widgets belong to the window mixins in\n'
         'measurements_viewer/main_window/. The behavior of these functions '
         'is\n'
         'specified in the docstring of measurements_viewer/__init__.py.', 1),
        ("the launcher's ``myutils.Logger``", "the launcher's ``app_info.Logger``",
         1),
        ('(`apps.CombineMetricsEquationDialog.evaluateCellEquation`)',
         '(`dialogs_combine_measurements.CombineMetricsEquationDialog.'
         'evaluateCellEquation`)', 1),
        GUIWIN + (1,),
    ], []),
    ('cellacdc/models/_env_config.py', [
        ('accepted by ``myutils.check_install_package``',
         'accepted by ``installs.check_install_package``', 1),
        ('# `myutils._CONVPAINT_PACKAGES`).',
         '# `model_installs._CONVPAINT_PACKAGES`).', 1),
    ], []),
    ('cellacdc/myutils/app_info.py', [
        ('#     qt_handler = widgets.QtHandler()',
         '#     qt_handler = gui_widgets_main_window_items.QtHandler()', 1),
    ], []),
    ('cellacdc/myutils/installs.py', [
        ('For *args and **kwargs see `myutils.check_install_package`.',
         'For *args and **kwargs see `check_install_package`.', 1),
    ], []),
    ('cellacdc/myutils/segm_models.py', [
        ('#     paramsWin = apps.QDialogModelParams(',
         '#     paramsWin = dialogs_model_params.QDialogModelParams(', 1),
    ], []),
    ('cellacdc/preprocess.py', [
        ('These functions are automatically added to `apps.QDialogModelParams` '
         'and they ',
         'These functions are automatically added to the model parameters '
         'dialog\n(`dialogs_model_params.QDialogModelParams`) and they ', 1),
    ], []),
    ('cellacdc/segmentation_module/main_window.py', [
        ('# Check if we should launch dataPrep:',
         '# Check if we should launch the data prep module:', 1),
        ('#      with dataPrep\n', '#      with the data prep module\n', 1),
    ], []),
    ('cellacdc/service/model_execution.py', [
        ('* the segmentation module (`segm.segmWin`),',
         '* the segmentation module (`segm_window.segmWin`),', 1),
        ('* the main GUI (`gui.guiWin.repeatSegm`,',
         '* the main GUI (`guiWin.repeatSegm`,', 1),
        ('via `workers.SegForLostIDsWorker`,',
         'via `gui_workers_segmentation_tracking.SegForLostIDsWorker`,', 1),
        ('``myutils.check_gpu_available`` after ``prepare()``.',
         '``model_installs.check_gpu_available`` after ``prepare()``.', 1),
    ], []),
    ('cellacdc/shared/dialogs/model_selection.py', [
        ('`_torch_utils.get_torch_options`', '`torch_utils.get_torch_options`',
         1),
    ], []),
    ('cellacdc/shared/dialogs/preprocessing.py', [
        ('# `widgets.PreProcessingSelector.askSetParams`)',
         '# `widgets_combo_boxes.PreProcessingSelector.askSetParams`)', 1),
        ("# self.cancelButton = widgets.cancelPushButton('Cancel')",
         "# self.cancelButton = widgets_push_buttons.cancelPushButton('Cancel')",
         1),
    ], []),
    ('cellacdc/trackers/CellACDC_2steps/CellACDC_2steps_tracker.py', [
        ('signals : cellacdc.workers.signals, optional',
         'signals : cellacdc.shared.workers.base.signals, optional', 1),
    ], []),
    ('cellacdc/utils/applyTrackFromTrackMateXML.py', [
        ('# win = apps.ApplyTrackTableSelectColumnsDialog(df, parent=self)',
         '# win = utils_applyTrackFromTable.ApplyTrackTableSelectColumnsDialog('
         'df, parent=self)', 1),
    ], []),
    ('cellacdc/utils/compute.py', [
        ('#         msg = widgets.myMessageBox(wrapText=False)',
         '#         msg = widgets_message_box.myMessageBox(wrapText=False)', 1),
    ], []),
    ('cellacdc/utils/computeBackgroundValues.py', [
        ('holds the keys documented in `workers.ComputeBackgroundValuesWorker`.',
         'holds the keys documented in `ComputeBackgroundValuesWorker`.', 1),
    ], []),
    ('cellacdc/whitelist.py', [
        ('debug with a lot of prints, also in gui.py, by default False',
         'debug with a lot of prints, also in the main GUI, by default False',
         1),
    ], []),
]

def _qualname_at(tree, line):
    """`Class.method` / `function` of the fingerprint item holding `line`
    (code nested in a function belongs to that function), or None."""
    parts = []
    body = tree.body
    while True:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and \
                    node.lineno <= line <= node.end_lineno:
                parts.append(node.name)
                if isinstance(node, ast.ClassDef):
                    body = node.body
                    break
                return '.'.join(parts)
        else:
            return '.'.join(parts) or None


def edited_items(repo, rel, old):
    """`<rel>::<Qual>` of every item whose text holds `old` (docstrings and
    comments; the fingerprint compare excuses exactly these items), or
    `<rel>::__doc__` for text outside functions and classes."""
    with open(os.path.join(repo, rel), 'rb') as file:
        text = file.read().decode('utf-8').replace('\r\n', '\n')
    tree = ast.parse(text)
    items, start = [], text.find(old)
    while start != -1:
        qual = _qualname_at(tree, text.count('\n', 0, start) + 1)
        item = f'{rel}::{qual}' if qual else f'{rel}::__doc__'
        if item not in items:
            items.append(item)
        start = text.find(old, start + 1)
    return items


def main(repo):
    for rel, pairs, items in EDITS:
        for old, new, count in pairs:
            found = edited_items(repo, rel, old)
            logged = list(items) + [i for i in found if i not in items]
            replace(repo, rel, old, new, count=count,
                    items=logged or [rel + '::__doc__'])


run(main)
