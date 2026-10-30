from unittest.mock import Mock
from pathlib import Path

import pytest

from cellacdc.service import setup_models


def test_preselected_models_use_isolated_setup(monkeypatch):
    setup = Mock(return_value='model-python')
    monkeypatch.setattr(setup_models, 'get_or_create_model_env', setup)
    setup_models.main(['cellpose3', 'cellpose_v3', 'stardist', '--yes'])
    assert [call.args[0] for call in setup.call_args_list] == [
        'cellpose_v3', 'StarDist',
    ]
    for call in setup.call_args_list:
        assert call.kwargs['auto_install'] is True
        assert call.kwargs['prompt_func'].__self__.__class__.__name__ == 'AutoHandler'


def test_setup_errors_propagate(monkeypatch):
    setup = Mock(side_effect=RuntimeError('installation failed'))
    monkeypatch.setattr(setup_models, 'get_or_create_model_env', setup)
    with pytest.raises(RuntimeError, match='installation failed'):
        setup_models.main(['cellpose', '--yes'])


def test_setup_command_is_registered():
    tomllib = pytest.importorskip('tomllib')
    path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    assert data['project']['scripts']['acdc-setup-models'] == (
        'cellacdc.service.setup_models:main'
    )
