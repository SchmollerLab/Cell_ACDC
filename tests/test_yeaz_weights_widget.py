import builtins
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cellacdc import myutils


def _guard_weights_registry(monkeypatch, tmp_path):
    from cellacdc.models import _yeaz_v2_weights

    monkeypatch.setattr(
        myutils, 'get_model_path',
        lambda *args, **kwargs: (None, str(tmp_path)),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ('torch', 'yeaz', 'cellacdc.models.YeaZ_v2') or name.startswith(
            ('torch.', 'yeaz.', 'cellacdc.models.YeaZ_v2.')
        ):
            raise AssertionError(f'GUI weights registry imported {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(
        myutils, 'check_install_yeaz',
        Mock(side_effect=AssertionError('GUI must not install YeaZ')),
    )
    return importlib.reload(_yeaz_v2_weights)


def test_yeaz_weights_registry_roundtrip(monkeypatch, tmp_path):
    registry = _guard_weights_registry(monkeypatch, tmp_path)
    names, paths = registry.load_models_filepath()
    assert names == ['Phase contrast', 'Bright-field', 'Fission yeast']
    assert paths == {
        'Phase contrast': str(tmp_path / 'weights_budding_PhC_multilab_0_1'),
        'Bright-field': str(tmp_path / 'weights_budding_BF_multilab_0_1'),
        'Fission yeast': str(tmp_path / 'weights_fission_multilab_0_2'),
    }
    custom_path = str(tmp_path / 'custom.pt')
    registry.add_model_filepath('Custom', custom_path)
    registry.add_model_filepath('Another', custom_path)
    names, paths = registry.load_models_filepath()
    assert names[-2:] == ['Custom', 'Another']
    assert paths['Custom'] == custom_path
    assert paths['Another'] == custom_path
    saved = tmp_path / registry.CUSTOM_WEIGHTS_JSON_FILENAME
    assert json.loads(saved.read_text()) == {
        'Custom': custom_path, 'Another': custom_path,
    }


def test_yeaz_widget_does_not_load_model_dependencies(
        monkeypatch, tmp_path,
    ):
    pytest.importorskip('qtpy')
    from qtpy.QtWidgets import QApplication
    from cellacdc import widgets

    app = QApplication.instance() or QApplication([])
    # Import the GUI first, then guard all registry and widget operations.
    with monkeypatch.context() as patch:
        registry = _guard_weights_registry(patch, tmp_path)
        registry.add_model_filepath('Existing', str(tmp_path / 'existing.pt'))
        widget = widgets.YeazV2SelectModelNameCombobox()
        try:
            assert [widget.itemText(i) for i in range(widget.count())] == [
                'Phase contrast', 'Bright-field', 'Fission yeast', 'Existing',
                'Select custom weights file...',
            ]
            custom_path = str(tmp_path / 'selected.pt')
            patch.setattr(
                widgets.qtpy.compat, 'getopenfilename',
                lambda **kwargs: (custom_path, ''),
            )
            patch.setattr(myutils, 'getMostRecentPath', lambda: str(tmp_path))
            patch.setattr(
                widgets.apps, 'QLineEditDialog',
                lambda **kwargs: SimpleNamespace(
                    cancel=False, enteredValue='Selected', exec_=lambda: None,
                ),
            )
            widget.setCurrentText('Select custom weights file...')
            assert widget.currentText() == 'Selected'
            names, paths = registry.load_models_filepath()
            assert names[-1] == 'Selected'
            assert paths['Selected'] == custom_path
            myutils.check_install_yeaz.assert_not_called()
        finally:
            widget.close()
            widget.deleteLater()
            app.processEvents()
