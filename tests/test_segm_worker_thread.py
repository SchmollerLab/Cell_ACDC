import os
from collections import namedtuple


ArgSpec = namedtuple('ArgSpec', ['name', 'type'])

def test_parse_model_params_keeps_explicit_none():
    from cellacdc import myutils

    argspecs = [
        ArgSpec('model_type', str),
        ArgSpec('model_path', os.PathLike),
        ArgSpec('batch_size', int),
        ArgSpec('augment', bool),
        ArgSpec('diameter', float),
    ]
    parsed = myutils.parse_model_params(argspecs, {
        'model_type': None, 'model_path': 'model.pt', 'batch_size': '36',
        'augment': None, 'unknown': 'value',
    })

    # Explicit None is passed to the model (not replaced by its default),
    # missing and unknown parameters are skipped
    assert parsed == {
        'model_type': None, 'model_path': 'model.pt', 'batch_size': 36,
        'augment': None,
    }


def test_workflow_config_roundtrip_keeps_model_type_none(tmp_path):
    """Running a workflow saved from the segmentation module with a custom
    cellpose `model_path` failed with "You cannot set both `model_type` and
    `model_path`", because `model_type = None` was dropped and the model
    default (`model_type='cpsam'`) was used instead.
    """
    from cellacdc import load, myutils

    images_path = tmp_path / 'Position_1' / 'Images'
    images_path.mkdir(parents=True)
    ini_path = tmp_path / 'workflow.ini'
    init_kwargs = {
        'model_type': None, 'model_path': 'C:/models/custom_model',
        'device_type': 'gpu', 'device': None, 'batch_size': 36,
    }
    load.save_workflow_to_config(
        str(ini_path), {'init_segmentation_model_params': init_kwargs},
        [str(images_path)], [1]
    )
    workflow_params = load.read_segm_workflow_from_config(str(ini_path))

    argspecs = [
        ArgSpec('model_type', str),
        ArgSpec('model_path', os.PathLike),
        ArgSpec('device_type', str),
        ArgSpec('device', str),
        ArgSpec('batch_size', int),
    ]
    parsed = myutils.parse_model_params(
        argspecs, workflow_params['init_segmentation_model_params']
    )
    assert parsed == init_kwargs
