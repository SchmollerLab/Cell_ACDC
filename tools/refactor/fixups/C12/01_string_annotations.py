"""C12: three string annotations name the old widgets module, which the
rewrite drops from their modules (flake8 F821; rewrite.py reports them as
`string_annotation`). Point them at the new homes: load.py gets a
TYPE_CHECKING import (no new runtime import edge), the two shared dialogs
already import the home module or define the class."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import ensure_line_after, has_line, replace, run  # noqa: E402

LOAD = 'cellacdc/load.py'
TC_IMPORT = '    from cellacdc.shared.widgets import plot_items as widgets_plot_items'


def main(repo):
    replace(repo, LOAD,
            "roiItem: 'widgets.PlotCurveItem',",
            "roiItem: 'widgets_plot_items.PlotCurveItem',",
            items=[LOAD + '::loadData.saveDataPrepFreeRoi'])
    ensure_line_after(repo, LOAD, 'from typing import List',
                      'from typing import TYPE_CHECKING')
    if not has_line(repo, LOAD, TC_IMPORT):
        replace(repo, LOAD,
                "\nacdc_df_bool_cols = [\n",
                "\nif TYPE_CHECKING:\n" + TC_IMPORT + "\n\nacdc_df_bool_cols = [\n",
                items=[LOAD + '::stmt:if TYPE_CHECKING:'])
    rel = 'cellacdc/shared/dialogs/preprocessing.py'
    replace(repo, rel,
            "selector: 'widgets.PreProcessingSelector'=None",
            "selector: 'widgets_combo_boxes.PreProcessingSelector'=None",
            items=[rel + '::PreProcessParamsWidget.setParamsStep'])
    rel = 'cellacdc/shared/dialogs/set_measurements.py'
    replace(repo, rel,
            "'Generator[Tuple[widgets._metricsQGBox, str], None, None]'",
            "'Generator[Tuple[_metricsQGBox, str], None, None]'",
            items=[rel + '::SetMeasurementsDialog.measurementSearchGroups'])


run(main)
