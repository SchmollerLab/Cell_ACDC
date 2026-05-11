import types

import numpy as np
from skimage.measure import regionprops
from types import SimpleNamespace

from cellacdc import myutils
from cellacdc.regionprops import acdcRegionprops
from cellacdc.workers import SegForLostIDsWorker

from cellacdc.models.thresholding.acdcSegment import Model as ThresholdingModel
from cellacdc.segm_utils import get_best_overlapping_label


def test_get_best_overlapping_label_uses_majority_overlap_with_allowed_labels():
    label_img = np.zeros((8, 8), dtype=np.uint16)
    label_img[2:6, 2:4] = 4
    label_img[3:7, 4:6] = 7

    obj = types.SimpleNamespace(
        slice=(slice(2, 7), slice(2, 6)),
        image=np.array(
            [
                [False, False, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, True],
            ]
        ),
    )

    assert get_best_overlapping_label(label_img, obj, {4, 7}) == 4


def test_get_best_overlapping_label_returns_none_without_allowed_overlap():
    label_img = np.zeros((6, 6), dtype=np.uint16)
    label_img[1:3, 1:3] = 2

    obj = types.SimpleNamespace(
        slice=(slice(1, 4), slice(1, 4)),
        image=np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ]
        ),
    )

    assert get_best_overlapping_label(label_img, obj, {5}) is None


def test_thresholding_model_object_can_be_mapped_back_to_missing_id():
    prev_lab = np.zeros((10, 10), dtype=np.uint16)
    prev_lab[3:7, 3:7] = 5

    image = np.zeros((10, 10), dtype=np.float32)
    image[3:7, 3:7] = 10.0

    model = ThresholdingModel()
    model_lab = model.segment(
        image,
        gauss_sigma=0,
        threshold_method='threshold_otsu',
    )

    rp_model = regionprops(model_lab)
    assert len(rp_model) == 1

    recovered_id = get_best_overlapping_label(prev_lab, rp_model[0], {5})

    assert recovered_id == 5


class _DummyLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class _DummySignal:
    def emit(self, *args, **kwargs):
        pass


class _DummySignals:
    def __init__(self):
        self.progress = _DummySignal()
        self.finished = _DummySignal()
        self.initProgressBar = _DummySignal()
        self.progressBar = _DummySignal()
        self.critical = _DummySignal()


def test_seg_for_lost_ids_worker_thresholding_relabels_recovered_object(monkeypatch):
    prev_lab = np.zeros((10, 10), dtype=np.uint16)
    prev_lab[3:7, 3:7] = 5

    curr_lab = np.zeros((10, 10), dtype=np.uint16)
    curr_img = np.zeros((10, 10), dtype=np.float32)
    curr_img[3:7, 3:7] = 10.0

    prev_rp = acdcRegionprops(prev_lab)
    curr_rp = acdcRegionprops(curr_lab)

    posData = SimpleNamespace(
        frame_i=1,
        lab=curr_lab.copy(),
        rp=curr_rp,
        allData_li=[
            {'labels': prev_lab, 'regionprops': prev_rp},
            {'labels': curr_lab, 'regionprops': curr_rp},
        ],
    )

    guiWin = SimpleNamespace(
        data=[posData],
        pos_i=0,
        SegForLostIDsSettings={
            'models_settings': [
                {
                    'base_model_name': 'thresholding',
                    'init_kwargs_new': {},
                    'args_new': {
                        'distance_filler_growth': 1.0,
                        'overlap_threshold': 0.5,
                            'padding': 1.0,
                        'size_perc_diff': 1.0,
                        'allow_only_tracked_cells': True,
                    },
                    'init_kwargs': {},
                    'model_kwargs': {
                        'gauss_sigma': 0,
                        'threshold_method': 'threshold_otsu',
                    },
                    'preproc_recipe': None,
                    'applyPostProcessing': False,
                    'standardPostProcessKwargs': {},
                    'customPostProcessFeatures': None,
                    'customPostProcessGroupedFeatures': None,
                }
            ]
        },
        getDisplayedImg1=lambda: curr_img,
        get_2Dlab=lambda lab: lab,
        getTrackedLostIDs=lambda: [],
        setBrushID=lambda useCurrentLab=True, return_val=True: 10,
        logger=_DummyLogger(),
    )

    worker = SegForLostIDsWorker(guiWin, mutex=SimpleNamespace(lock=lambda: None, unlock=lambda: None), waitCond=SimpleNamespace(wait=lambda mutex: None))
    worker.signals = _DummySignals()
    worker.logger = _DummyLogger()
    worker.gpu_go = True
    worker.dont_force_cpu = True

    monkeypatch.setattr(worker, 'emitSigAskInit', lambda: None)
    monkeypatch.setattr(worker, 'emitSigAskInstallGPU', lambda base_model_name, use_gpu: None)
    monkeypatch.setattr(worker, 'emitSigUpdateRP', lambda wl_update=True, wl_track_og_curr=False: None)
    monkeypatch.setattr(worker, 'emitSigStoreData', lambda autosave=True: None)
    monkeypatch.setattr(worker, 'emitTrackManuallyAddedObject', lambda *args, **kwargs: None)
    monkeypatch.setattr(myutils, 'import_segment_module', lambda base_model_name: SimpleNamespace(Model=ThresholdingModel))
    monkeypatch.setattr(myutils, 'init_segm_model', lambda acdcSegment, posData, init_kwargs_new: ThresholdingModel())

    worker.run()

    assert posData.lab[3:7, 3:7].min() == 5
    assert posData.lab[3:7, 3:7].max() == 5