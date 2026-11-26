# Test importing the cellacdc modules.

import pytest
import cellacdc

from cellacdc import gui
from cellacdc import dataStruct
from cellacdc import segm
from cellacdc import dataPrep

pytestmark = pytest.mark.infra


@pytest.mark.smoke
def test_placeholder():
    # Add test test_placeholder because we are only testing import
    pass
