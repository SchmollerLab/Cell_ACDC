# Test importing the cellacdc modules.

import pytest
import cellacdc

from cellacdc.gui.main_window import window as gui
from cellacdc.data_structure import data_struct as dataStruct
from cellacdc.segmentation_module import main_window as segm
from cellacdc.data_prep import main_window as dataPrep

pytestmark = pytest.mark.infra


@pytest.mark.smoke
def test_placeholder():
    # Add test test_placeholder because we are only testing import
    pass
