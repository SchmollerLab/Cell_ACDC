"""Expected GUI-visible changes of Module 0 compared with main.

Registry used by `test_golden_module0.py`: one `ExpectedChange` per change
id (GV-01 ... GV-20) that the refactor makes visible in a golden scenario.
`assert_golden` requires every golden diff to be claimed by an entry listed
for that scenario and every listed entry to claim at least one diff, so this
list is the reviewed list of GUI changes of the PR.

Created empty on main (the goldens were captured there); entries are added
together with the change that causes them.
"""
from __future__ import annotations

import math

from . import manifest
from .scenarios import ExpectedChange, assert_golden

__all__ = ['CHANGES', 'ExpectedChange', 'assert_golden']


def _none_text_became_metadata(diff: manifest.Diff) -> bool:
    return diff.old is True and diff.new is False


def _nm_value_now_in_um(diff: manifest.Diff) -> bool:
    try:
        return math.isclose(float(diff.new), float(diff.old) / 1000)
    except (TypeError, ValueError):
        return False


CHANGES: list[ExpectedChange] = [
    # Comes with the reader fix (`Metadata.init_from_file` returns self), so
    # the legacy worker already shows it. Scenarios = every scenario that
    # writes a Position from a non-'.ome.tif' file through the legacy worker.
    ExpectedChange(
        gv='GV-01',
        scenarios=(
            'S01', 'S02', 'S03', 'S04', 'S05', 'S08', 'S09', 'S10', 'S11',
            'S12a', 'S12b', 'S13', 'S14', 'S18a', 'S18b',
        ),
        key_glob='dst/*_metadataXML.txt/is_literal_None',
        predicate=_none_text_became_metadata,
        description=(
            '*_metadataXML.txt for non-.ome.tif input holds the reader '
            'metadata (str(bioimage.metadata)) instead of the text None.'
        ),
    ),
    # Metadata dialog (`convertUnits`); S17 shows nm pixel sizes.
    ExpectedChange(
        gv='GV-03',
        scenarios=('S17',),
        key_glob='dst/*_metadata.csv/rows/*/1',
        predicate=_nm_value_now_in_um,
        description=(
            'Pixel sizes shown/entered in nm, mm or cm are converted to um '
            'before metadata.csv and TIFF spacing (previously written '
            'unconverted).'
        ),
    ),
]
