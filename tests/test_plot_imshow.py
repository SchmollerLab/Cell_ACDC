"""``cellacdc.plot.imshow``: the debugging image viewer (``ImShow`` of
``shared/widgets/imshow_window.py``) opens offscreen without blocking."""
import numpy as np
import pytest
from qtpy.QtWidgets import QApplication

from cellacdc import plot
from cellacdc.shared.widgets import imshow_window

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_imshow_shows_images_without_blocking(app):
    images = np.zeros((3, 8, 8), dtype=np.uint8)
    labels = np.zeros((3, 8, 8), dtype=np.uint32)
    labels[:, 2:5, 2:5] = 1
    win = plot.imshow(images, labels, block=False, win_stay_on_top=False)
    try:
        assert isinstance(win, imshow_window.ImShow)
        assert win.isVisible()
    finally:
        win.close()
        win.deleteLater()
