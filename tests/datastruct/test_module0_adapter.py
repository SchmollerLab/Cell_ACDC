"""`dataStruct.bioFormatsWorker` as a thin adapter of `convert()`.

The worker runs synchronously in this thread (fake mutex and wait
condition, no QThread) with `InProcessBackend(FakeBioImage)` and every
`confirmMetadata` answered by a scripted closed dialog:

- outcome mapping of `run`: `MetadataReadError` -> `criticalError` +
  `finished`; `MoveRawFilesError` with a permission error ->
  `sigMoveRawMicrFilesFailed` and no `finished` (the GUI slot emits it); any
  other error -> `critical`, then `finished` (GV-19); nothing written but not
  cancelled -> "Conversion task ended"; Cancel -> "cancelled";
- `_confirm_settings`: the 17 `confirmMetadata` args (one file per channel:
  arg 0 is the first raw file), the mutex held from the emit to the
  wait, the eye re-read with the edited values, a failing sample read never
  locks the mutex (GV-19);
- `createDataStructWin.attemptSeparateMultiChannel` (GV-08) and `initBioIO`
  (reader check per file extension, then one `init_reader` task, GV-20);
- `test_child_process_count` and `test_parent_never_opens_raw_gui` through
  the production backend;
- the golden registry (`expected_module0_changes`) claims exactly the
  expected diffs: golden scenarios run in process (real bioio) fail when a
  registered output is wrong, missing or left over.
"""
from __future__ import annotations

import copy
import fnmatch
import os
from types import SimpleNamespace

import pytest
import tifffile

from cellacdc.acdc_datastruct import backends, naming, runner

from . import (
    conditions, expected_module0_changes, fake_bioio, harness_module0,
    scenarios
)
from .harness_module0 import CONFIRM_ARG_NAMES, FakeMutex, FakeWaitCondition

pytestmark = conditions.requires_qt

FACTORY = fake_bioio.FakeBioImage
METADATA_ERROR_ACTION = 'reading image data or metadata'


@pytest.fixture(scope='module', autouse=True)
def qapp():
    return conditions.qapp()


# ------------------------------------------------------------------ inputs
def fake_file(folder, name: str, **spec) -> str:
    """Fake raw file (T=3, C=2, Z=1, 4x5 pixels unless given)."""
    spec = {'T': 3, 'C': 2, 'Z': 1, 'Y': 4, 'X': 5, **spec}
    spec.setdefault('channel_names', ['GFP', 'mCherry', 'Cy5'][:spec['C']])
    fake_bioio.write_fake_raw(
        os.path.join(str(folder), name), fake_bioio.make_spec(**spec)
    )
    return name


def one_per_pos_files(folder, count: int = 3, **spec) -> list[str]:
    return [
        fake_file(folder, f'pos{i + 1}.czi', s_offset=i, **spec)
        for i in range(count)
    ]


def one_per_channel_files(folder) -> list[str]:
    return sorted(
        fake_file(folder, f'ASY015_{pos}_{channel}.czi', C=1, s_offset=pos,
                  channel_names=[channel])
        for pos in (1, 2) for channel in ('mNeon-Green', 'GFP')
    )


class SpyBackend(backends.InProcessBackend):
    """`InProcessBackend(FakeBioImage)` recording every task in `calls`.
    `fail` maps a task name to the exception it raises; `write=False` makes
    `save_positions` write nothing (no event)."""
    def __init__(self, *, write: bool = True, **fail) -> None:
        super().__init__(reader_factory=FACTORY)
        self.calls: list[tuple] = []
        self.fail = fail
        self.write = write

    def _record(self, task: str, *details) -> None:
        self.calls.append((task, *details))
        if task in self.fail:
            raise self.fail[task]

    def read_metadata(self, paths):
        self._record('read_metadata', *map(os.path.basename, paths))
        return super().read_metadata(paths)

    def read_sample(self, path, *, size_c, size_t, size_z, lazy_load):
        self._record('read_sample', (size_c, size_t, size_z))
        return super().read_sample(path, size_c=size_c, size_t=size_t,
                                   size_z=size_z, lazy_load=lazy_load)

    def save_positions(self, jobs, texts, *, on_event):
        self._record('save_positions', [job.input_id for job in jobs])
        if not self.write:
            return []
        return super().save_positions(jobs, texts, on_event=on_event)

    @property
    def tasks(self) -> list[str]:
        return [call[0] for call in self.calls]


# ------------------------------------------------------------ worker runs
def closed_dialog(args: dict, answer: dict) -> SimpleNamespace:
    """The attributes `QDialogMetadataXML.ok_cb` leaves for these confirm
    args (nothing edited, every channel saved, all positions), updated with
    `answer`."""
    size_c, size_t = args['SizeC'], args['SizeT']
    dialog = SimpleNamespace(
        cancel=False, requestedReadingSampleImageDataAgain=False,
        trust=False, overWrite=False, to_h5=False,
        LensNA=args['LensNA'], SizeT=size_t, SizeZ=args['SizeZ'],
        SizeC=size_c, SizeS=args['SizeS'],
        TimeIncrement=args['TimeIncrement'],
        PhysicalSizeX=args['PhysicalSizeX'],
        PhysicalSizeY=args['PhysicalSizeY'],
        PhysicalSizeZ=args['PhysicalSizeZ'],
        chNames=[naming.sanitize_name(name) for name in args['chNames']],
        saveChannels=[True] * size_c, emWavelens=list(args['emWavelens']),
        timeRangeToSave=(1, size_t), selectedPos=['All Positions'],
        addImageName=False,
    )
    vars(dialog).update(answer)
    return dialog


class RecordingMutex(FakeMutex):
    def __init__(self) -> None:
        self.locked = False
        self.locks = 0

    def lock(self) -> None:
        assert not self.locked, 'the mutex is already locked'
        self.locked = True
        self.locks += 1

    def unlock(self) -> None:
        assert self.locked, 'the mutex is not locked'
        self.locked = False


class CheckingWaitCondition(FakeWaitCondition):
    """`wait` is only called with the mutex locked (it releases it)."""
    def wait(self, mutex, *args) -> bool:
        assert mutex.locked, 'wait() without the mutex locked'
        return True


class WorkerRun:
    """A `bioFormatsWorker` run synchronously: every signal is recorded in
    `signals` (slots connected directly, so they run inside `emit`) and
    every `confirmMetadata` is answered with `closed_dialog(args, answer)`
    of the next scripted answer ({} = Ok)."""
    def __init__(self, src, files, dst, *, layout: int = 1, backend=None,
                 answers=(), **kwargs) -> None:
        from cellacdc import dataStruct

        self.answers = list(answers)
        self.mutex = RecordingMutex()
        self.signals: list[tuple] = []
        self.confirm_args: list[dict] = []
        self.locked_at_confirm: list[bool] = []
        self.pbar: list[tuple] = []
        kwargs.setdefault('move_raw_microscopy_files', False)
        worker = dataStruct.bioFormatsWorker(
            str(src), list(files), str(dst), self.mutex,
            CheckingWaitCondition(), layout, 'bioio', backend=backend,
            **kwargs
        )
        worker.useSymLink = False
        worker.progress.connect(self._on_progress)
        worker.initPbar.connect(lambda n: self.pbar.append(('init', n)))
        worker.progressPbar.connect(lambda n: self.pbar.append(('step', n)))
        worker.criticalError.connect(self._on_critical_error)
        worker.critical.connect(self._on_critical)
        worker.sigMoveRawMicrFilesFailed.connect(self._on_move_failed)
        worker.confirmMetadata.connect(self._on_confirm)
        worker.finished.connect(lambda: self.signals.append(('finished',)))
        self.worker = worker

    def _on_progress(self, text) -> None:
        self.signals.append(('progress', text))

    def _on_critical_error(self, action, details, filename) -> None:
        self.signals.append(
            ('criticalError', action, filename, bool(details.strip()))
        )

    def _on_critical(self, error) -> None:
        self.signals.append(('critical', type(error).__name__))

    def _on_move_failed(self, details) -> None:
        self.signals.append(('sigMoveRawMicrFilesFailed', bool(details)))

    def _on_confirm(self, *args) -> None:
        # Never raise here: an exception in a slot aborts the process
        self.locked_at_confirm.append(self.mutex.locked)
        self.signals.append(('confirmMetadata', len(self.confirm_args)))
        values = dict(zip(CONFIRM_ARG_NAMES, args))
        self.confirm_args.append(values)
        answer = self.answers.pop(0) if self.answers else {}
        self.worker.metadataWin = closed_dialog(values, answer)

    def run(self) -> 'WorkerRun':
        self.worker.run()
        assert not self.mutex.locked
        return self

    @property
    def names(self) -> list[str]:
        return [signal[0] for signal in self.signals]

    @property
    def task_ended(self) -> str:
        """What `createDataStructWin.taskEnded` shows after `finished`."""
        worker = self.worker
        if worker.cancelled and not worker.isCriticalError:
            return 'cancelled'
        if not worker.cancelled:
            return 'ended'
        return 'nothing'


def written_positions(dst) -> list[str]:
    if not os.path.isdir(dst):
        return []
    return sorted(name for name in os.listdir(dst)
                  if name.startswith('Position_'))


# --------------------------------------------------------- outcome mapping
def test_conversion_ends(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=2)
    backend = SpyBackend()
    run = WorkerRun(src, files, dst, backend=backend).run()
    assert run.names.count('confirmMetadata') == 2
    assert run.names[-1] == 'finished' and run.names.count('finished') == 1
    assert 'critical' not in run.names
    assert run.task_ended == 'ended'
    assert written_positions(dst) == ['Position_1', 'Position_2']
    # one dialog per file, before that file is written
    assert backend.tasks == [
        'read_metadata', 'read_sample', 'save_positions',
    ] * 2
    assert run.locked_at_confirm == [True, True]
    assert run.pbar == [('init', 4), ('step', 2), ('step', 2)]


def test_metadata_read_error_shows_the_bioformats_error(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = [fake_file(src, 'bad.czi', fail={'stage': 'open'})]
    run = WorkerRun(src, files, dst, backend=SpyBackend()).run()
    assert run.signals == [
        ('progress', 'Reading OME metadata...'),
        ('criticalError', METADATA_ERROR_ACTION, 'bad.czi', True),
        ('finished',),
    ]
    assert run.worker.isCriticalError and run.worker.cancelled
    assert run.task_ended == 'nothing'


def test_permission_error_moving_raw_files(tmp_path, monkeypatch):
    from cellacdc import io

    def refuse(src, dst_folder):
        raise PermissionError(13, 'Permission denied', src)

    monkeypatch.setattr(io, 'move_raw_microscopy_file', refuse)
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1)
    backend = SpyBackend()
    run = WorkerRun(src, files, dst, backend=backend,
                    move_raw_microscopy_files=True).run()
    # createDataStructWin.moveRawMicroscopyFilesFailed emits `finished`
    assert run.signals == [('sigMoveRawMicrFilesFailed', True)]
    assert run.worker.cancelled and not run.worker.isCriticalError
    assert backend.calls == []
    assert sorted(os.listdir(src)) == files


@pytest.mark.parametrize('failure', [
    'move', 'save', 'destination is a file', 'sample', 'case variants',
])
def test_other_errors_show_critical_then_finished(
        tmp_path, monkeypatch, failure):
    """GV-19: the error dialog (`critical`), then `finished` so that the
    thread quits; `taskEnded` shows nothing more. A failing sample read
    never locks the mutex. Raw files to move whose names differ only in
    case are rejected before anything is moved (GV-16)."""
    from cellacdc import io

    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1)
    kwargs = {}
    fail = {}
    if failure == 'move':
        def fail_move(src, dst_folder):
            raise OSError(5, 'Input/output error', src)
        monkeypatch.setattr(io, 'move_raw_microscopy_file', fail_move)
        kwargs['move_raw_microscopy_files'] = True
        expected = 'MoveRawFilesError'
    elif failure == 'save':
        fail['save_positions'] = RuntimeError('disk full')
        expected = 'RuntimeError'
    elif failure == 'destination is a file':
        dst.write_text('not a folder')
        expected = 'PlanError'
    elif failure == 'case variants':
        # One file on a case-insensitive file system, two otherwise
        files = [fake_file(src, 'Pos1.czi'), 'pos1.czi']
        kwargs['move_raw_microscopy_files'] = True
        expected = 'PlanError'
    else:
        fail['read_sample'] = RuntimeError('cannot read the sample')
        expected = 'RuntimeError'
    run = WorkerRun(src, files, dst, backend=SpyBackend(**fail),
                    **kwargs).run()
    assert run.signals[-2:] == [('critical', expected), ('finished',)]
    assert run.names.count('finished') == 1
    assert run.worker.isCriticalError and run.worker.cancelled
    assert run.task_ended == 'nothing'
    if failure == 'sample':
        assert run.mutex.locks == 0
        assert 'confirmMetadata' not in run.names
    if failure == 'case variants':
        assert run.signals == [('critical', 'PlanError'), ('finished',)]
        assert not dst.exists()


def test_nothing_written_still_ends(tmp_path):
    """The GUI ignores ConvertResult.ok's "at least one Position
    written": not cancelled -> "Conversion task ended"."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1)
    run = WorkerRun(src, files, dst, backend=SpyBackend(write=False)).run()
    assert run.names[-1] == 'finished'
    assert 'critical' not in run.names
    assert run.task_ended == 'ended'
    assert written_positions(dst) == []


def test_cancel_in_the_dialog(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=2)
    backend = SpyBackend()
    run = WorkerRun(src, files, dst, backend=backend,
                    answers=[{'cancel': True}]).run()
    assert run.names.count('confirmMetadata') == 1
    assert run.names[-1] == 'finished'
    assert run.task_ended == 'cancelled'
    assert 'save_positions' not in backend.tasks
    assert written_positions(dst) == []


# --------------------------------------------------------- confirm adapter
def test_confirm_args(tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1, T=4, C=2, Z=3)
    run = WorkerRun(src, files, dst, backend=SpyBackend()).run()
    args = run.confirm_args[0]
    assert {key: args[key] for key in (
        'filename', 'SizeT', 'SizeZ', 'SizeC', 'SizeS', 'chNames',
        'rawFilePath'
    )} == {
        'filename': 'pos1.czi', 'SizeT': 4, 'SizeZ': 3, 'SizeC': 2,
        'SizeS': 1, 'chNames': ['GFP', 'mCherry'],
        'rawFilePath': os.path.join(str(src), 'pos1.czi'),
    }
    sample = args['sampleImgData']
    assert sorted(sample) == sorted(['zct', 'ztc', 'czt', 'ctz', 'tzc', 'tcz'])
    assert [array.shape for array in sample['ctz']] == [(4, 3, 4, 5)] * 2
    progress = [s[1] for s in run.signals if s[0] == 'progress']
    assert progress.index('Reading sample image data...') < (
        run.names.index('confirmMetadata')
    )


def test_one_file_per_channel_confirm_args(tmp_path):
    """Arg 0 is the first raw file (askConfirmMetadata shows the
    basename); channel names are the natural-sorted raw tokens; no sample
    data."""
    from cellacdc import dataStruct

    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_channel_files(src)
    stub = SimpleNamespace(logger=None, criticalNoFilenamePattern=None)
    assert dataStruct.createDataStructWin.attemptSeparateMultiChannel(
        stub, files
    )
    run = WorkerRun(src, files, dst, layout=2, backend=SpyBackend())
    for name in ('one_per_channel', 'basename', 'SizeS', 'posNums',
                 'chNames'):
        setattr(run.worker, name, getattr(stub, name))
    run.run()
    args = run.confirm_args[0]
    assert args['filename'] == files[0] == 'ASY015_1_GFP.czi'
    assert args['SizeC'] == 2 and args['SizeS'] == 2
    assert args['chNames'] == ['GFP', 'mNeon-Green']
    assert args['emWavelens'] == [500.0, 500.0]
    assert args['sampleImgData'] is None
    assert ('progress', 'Reading sample image data...') not in run.signals
    assert written_positions(dst) == ['Position_1', 'Position_2']
    assert sorted(os.listdir(dst / 'Position_1' / 'Images')) == [
        'ASY015__s1_GFP.tif', 'ASY015__s1_mNeon_Green.tif',
        'ASY015__s1_metadata.csv', 'ASY015__s1_metadataXML.txt',
    ]


def test_eye_reread_shows_the_edited_values(tmp_path):
    """The preview button after editing SizeT/SizeZ closes the dialog: the
    sample is read again with the edited sizes and the dialog shows the
    edited (not unit-converted) values; Ok then uses them (S14)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1, Z=2)
    backend = SpyBackend()
    reread = {
        'requestedReadingSampleImageDataAgain': True, 'SizeT': 2,
        'SizeZ': 1, 'TimeIncrement': 2.5, 'chNames': ['A', 'B'],
    }
    run = WorkerRun(src, files, dst, backend=backend,
                    answers=[reread]).run()
    assert [call for call in backend.calls if call[0] == 'read_sample'] == [
        ('read_sample', (2, 3, 2)), ('read_sample', (2, 2, 1)),
    ]
    second = run.confirm_args[1]
    assert (second['SizeT'], second['SizeZ'], second['TimeIncrement']) == (
        2, 1, 2.5
    )
    assert second['chNames'] == ['A', 'B']
    assert [array.shape for array in second['sampleImgData']['zct']] == [
        (2, 4, 5)
    ] * 2
    images = dst / 'Position_1' / 'Images'
    csv = (images / 'pos1_s1_metadata.csv').read_text()
    assert 'SizeT,2' in csv and 'SizeZ,1' in csv
    assert sorted(os.listdir(images))[:2] == ['pos1_s1_A.tif', 'pos1_s1_B.tif']


def test_trust_asks_again_for_a_file_with_other_channels(tmp_path):
    """The "Trust metadata reader" button skips the dialog of a compatible
    file and shows it again for a file with another number of channels."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=2) + [
        fake_file(src, 'pos3.czi', C=3, s_offset=2)
    ]
    run = WorkerRun(src, files, dst, backend=SpyBackend(),
                    answers=[{'trust': True}]).run()
    assert [args['filename'] for args in run.confirm_args] == [
        'pos1.czi', 'pos3.czi'
    ]
    assert run.task_ended == 'ended'
    assert written_positions(dst) == ['Position_1', 'Position_2',
                                       'Position_3']


def test_reuse_with_the_full_range_writes_all_frames_of_later_files(
        tmp_path):
    """GV-07: "Use the above metadata for all the next positions" with the
    full time range of file 1 (3 frames) writes all 5 frames of file 2
    (main wrote file 1's range, frames 1-3, and SizeT 3)."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = [fake_file(src, 'pos1.czi', T=3),
             fake_file(src, 'pos2.czi', T=5, s_offset=1)]
    run = WorkerRun(src, files, dst, backend=SpyBackend(),
                    answers=[{'overWrite': True}]).run()
    assert [args['filename'] for args in run.confirm_args] == ['pos1.czi']
    assert run.task_ended == 'ended'
    for position, size_t in ((1, 3), (2, 5)):
        images = dst / f'Position_{position}' / 'Images'
        stem = f'pos{position}_s{position}_'
        for channel in ('GFP', 'mCherry'):
            frames = tifffile.imread(images / f'{stem}{channel}.tif')
            assert frames.shape[0] == size_t
        rows = (images / f'{stem}metadata.csv').read_text().splitlines()
        assert f'SizeT,{size_t}' in rows


# ----------------------------------------------------- createDataStructWin
def test_ask_confirm_metadata_wakes_the_worker_under_the_lock(monkeypatch):
    """GUI side of the hand-shake: after the dialog closed, `metadataWin`
    is handed over and the worker woken while holding the mutex (the
    worker holds it until it waits, so the wake-up cannot be lost)."""
    from cellacdc import apps, dataStruct

    calls = []

    class Dialog:
        def __init__(self, **kwargs) -> None:
            calls.append(('dialog', kwargs['title'], kwargs['rawDataStruct']))

        def exec_(self) -> None:
            calls.append(('exec_',))

    class Mutex:
        def lock(self):
            calls.append(('lock',))

        def unlock(self):
            calls.append(('unlock',))

    class WaitCondition:
        def wakeAll(self):
            calls.append(('wakeAll', type(stub.worker.metadataWin).__name__))

    monkeypatch.setattr(apps, 'QDialogMetadataXML', Dialog)
    stub = SimpleNamespace(
        rawDataStruct=2, basename='ASY015_', mutex=Mutex(),
        waitCond=WaitCondition(), worker=SimpleNamespace(metadataWin=None),
    )
    args = ['ASY015_1_GFP.czi', 1.4, 3, 1, 2, 2, 1.0, 's', 1.0, 1.0, 1.0,
            'μm', ['GFP', 'mCherry'], [500.0, 500.0], '', 'path', None]
    dataStruct.createDataStructWin.askConfirmMetadata(stub, *args)
    assert calls == [
        ('dialog', 'Metadata for ASY015_', 2), ('exec_',), ('lock',),
        ('wakeAll', 'Dialog'), ('unlock',),
    ]
    assert stub.metadataDialogIsOpen is False


@pytest.mark.parametrize('files, basename, positions, channels', [
    (['ASY015_1_GFP.tif', 'ASY015_1_mNeon.tif', 'ASY015_2_GFP.tif'],
     'ASY015_', [1, 2], ['GFP', 'mNeon']),
    (['exp_01_RFP.ome.tif', 'exp_10_GFP.ome.tif', 'exp_01_GFP.ome.tif'],
     'exp_', [1, 10], ['GFP', 'RFP']),
    (['a_1_ch10.tif', 'a_1_ch2.tif'], 'a_', [1], ['ch2', 'ch10']),
])
def test_separate_multichannel_names(files, basename, positions, channels):
    """GV-08: natural-sorted raw tokens ('.ome' removed), zero padding."""
    from cellacdc import dataStruct

    stub = SimpleNamespace(logger=None, criticalNoFilenamePattern=None)
    assert dataStruct.createDataStructWin.attemptSeparateMultiChannel(
        stub, files
    ) is True
    assert stub.basename == basename == stub.one_per_channel.basename
    assert stub.posNums == positions
    assert stub.SizeS == len(positions)
    assert stub.chNames == channels


def test_separate_multichannel_pattern_error():
    import logging

    from cellacdc import dataStruct

    boxes = []
    stub = SimpleNamespace(
        logger=logging.getLogger('test_module0_adapter'),
        criticalNoFilenamePattern=lambda error='': boxes.append(error),
    )
    assert dataStruct.createDataStructWin.attemptSeparateMultiChannel(
        stub, ['ASY015_1_GFP.tif', 'notes.tif']
    ) is False
    assert boxes == ['notes.tif']
    assert not hasattr(stub, 'one_per_channel')


@pytest.fixture
def installs(monkeypatch):
    """`install_reader_dependencies` replaced by a recorder."""
    from cellacdc.acdc_bioio_bioformats import install

    calls = []

    def record(path, exception, qparent=None, **kwargs):
        calls.append((os.path.basename(path), str(exception), qparent))

    monkeypatch.setattr(install, 'install_reader_dependencies', record)
    return calls


def test_initbioio_installs_per_extension(installs, monkeypatch, tmp_path):
    """GV-20: the reader of every extension is checked in the GUI
    process (the worker children cannot install); one init_reader task for
    the first file."""
    from cellacdc import dataStruct

    tasks = []

    def fake_run_task(task, params, **kwargs):
        tasks.append((task, dict(params)))
        return {'path': params['path']}

    monkeypatch.setattr(runner, 'run_task', fake_run_task)
    src = str(tmp_path)
    stub = SimpleNamespace(bioformats_backend='bioio')
    dataStruct.createDataStructWin.initBioIO(
        stub, src, ['a.czi', 'b.czi', 'c.nd2']
    )
    message = ('Failed installing reader dependencies from the GUI, '
               'trying from terminal...')
    assert installs == [('a.czi', message, stub), ('c.nd2', message, stub)]
    assert tasks == [('init_reader', {'path': os.path.join(src, 'a.czi')})]


def test_initbioio_python_bioformats_does_nothing(installs, monkeypatch):
    from cellacdc import dataStruct

    monkeypatch.setattr(runner, 'run_task', None)
    stub = SimpleNamespace(bioformats_backend='python-bioformats')
    dataStruct.createDataStructWin.initBioIO(stub, 'unused', ['a.czi'])
    assert installs == []


def test_child_process_count(installs, monkeypatch, tmp_path):
    """A GUI conversion of 3 files (one per position, no re-read)
    runs 1 init_reader + 3 x (read_metadata + read_sample +
    save_positions) = 10 worker tasks with the production backend (the
    runner is replaced by an in-process spy)."""
    from cellacdc import dataStruct

    in_process = backends.InProcessBackend(reader_factory=FACTORY)
    tasks = []

    def spy_run_task(task, params, *, on_event=None, collect=None,
                     **kwargs):
        tasks.append(task)
        return in_process._run(task, params, on_event=on_event,
                               collect=collect)

    monkeypatch.setattr(runner, 'run_task', spy_run_task)
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=3)
    dataStruct.createDataStructWin.initBioIO(
        SimpleNamespace(bioformats_backend='bioio'), str(src), files
    )
    run = WorkerRun(src, files, dst).run()        # default backend
    assert isinstance(run.worker.backend, backends.SubprocessBackend)
    assert run.task_ended == 'ended'
    assert tasks == ['init_reader'] + [
        'read_metadata', 'read_sample', 'save_positions'
    ] * 3
    assert len(tasks) == 10


@conditions.subprocess_heavy
def test_parent_never_opens_raw_gui(installs, monkeypatch, tmp_path):
    """With every raw-file open refused in this process, the GUI path
    (initBioIO + worker with the default backend) converts through worker
    children (FakeBioImage test hook)."""
    from cellacdc import dataStruct
    from cellacdc.acdc_bioio_bioformats import reader

    def refuse(*args, **kwargs):
        raise AssertionError('the GUI process opened a raw file')

    monkeypatch.setattr(reader.ImageReader, '__init__', refuse)
    monkeypatch.setattr(FACTORY, '__init__', refuse)
    try:
        import bioio
    except ImportError:
        pass
    else:
        monkeypatch.setattr(bioio.BioImage, '__init__', refuse)
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    monkeypatch.setenv(backends.READER_FACTORY_ENV,
                       fake_bioio.FAKE_BIOIMAGE_PATH)

    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1)
    dataStruct.createDataStructWin.initBioIO(
        SimpleNamespace(bioformats_backend='bioio'), str(src), files
    )
    run = WorkerRun(src, files, dst).run()
    assert 'critical' not in run.names, run.signals
    assert run.task_ended == 'ended'
    assert [array.shape for array in run.confirm_args[0]['sampleImgData'][
        'zct']] == [(3, 4, 5)] * 2
    assert sorted(os.listdir(dst / 'Position_1' / 'Images')) == [
        'pos1_s1_GFP.tif', 'pos1_s1_mCherry.tif', 'pos1_s1_metadata.csv',
        'pos1_s1_metadataXML.txt',
    ]


# ------------------------------------------- golden registry: exact claims
@pytest.fixture(scope='module')
def scenario_record(tmp_path_factory):
    """`scenario_record(id)`: a copy of the record of a Module 0 golden
    scenario run in process with real bioio (each scenario runs once)."""
    records = {}

    def record(scenario_id: str) -> dict:
        if scenario_id not in records:
            records[scenario_id] = harness_module0.run_module0_scenario(
                scenario_id, str(tmp_path_factory.mktemp(scenario_id)),
                backend=backends.InProcessBackend()
            )
        return copy.deepcopy(records[scenario_id])
    return record


def drop(pattern: str):
    """The destination files matching `pattern` are missing."""
    def mutate(record, golden) -> None:
        for key in fnmatch.filter(list(record['dst']), pattern):
            del record['dst'][key]
    return mutate


def copy_file(source: str, target: str):
    """Destination file `target` holds the content of `source`."""
    def mutate(record, golden) -> None:
        record['dst'][target] = copy.deepcopy(record['dst'][source])
    return mutate


def left_over(relpath: str):
    """Main's destination file `relpath` is still written."""
    def mutate(record, golden) -> None:
        record['dst'][relpath] = copy.deepcopy(golden['dst'][relpath])
    return mutate


def corrupt_pixels(relpath: str):
    def mutate(record, golden) -> None:
        record['dst'][relpath]['sha256'] = '0' * 64
    return mutate


def set_csv_row(relpath: str, name: str, value: str):
    def mutate(record, golden) -> None:
        for row in record['dst'][relpath]['rows']:
            if row[0] == name:
                row[1] = value
    return mutate


def replace_last_log_line(record, golden) -> None:
    progress = [s for s in record['signals'] if s[0] == 'progress']
    progress[-1][1] = 'Traceback (most recent call last):'


def drop_last_dialog(record, golden) -> None:
    del record['dialogs'][-1]


def asy015(p: int, name: str) -> str:
    """Output file of input position p of an 'ASY015_' layout-2 set."""
    return f'Position_{p}/Images/ASY015__s{p}_{name}'


def scene(k: int, name: str) -> str:
    """Output file of scene k of 'exp.ome.tif' with "Include image name"."""
    return f'Position_{k}/Images/exp_ome_Scene__{k}_s{k}_{name}'


def pos(n: int, name: str) -> str:
    """Output file of 'pos<n>.tif' (layout 1, one file per position)."""
    return f'Position_{n}/Images/pos{n}_s{n}_{name}'


@conditions.requires_bioio
@pytest.mark.parametrize('scenario_id, mutate', [
    pytest.param('S08', drop('Position_1/Images/*.tif'),
                 id='S08-position_1_images_missing'),
    pytest.param('S08', drop(asy015(1, 'notes.txt')),
                 id='S08-other_file_not_copied'),
    pytest.param('S08', left_over(asy015(1, 'x.txt')),
                 id='S08-file_of_position_10_still_copied'),
    pytest.param('S12b', copy_file(asy015(4, 'GFP.tif'),
                                   asy015(3, 'GFP.tif')),
                 id='S12b-position_3_holds_input_2'),
    pytest.param('S12b', drop('Position_4/*'), id='S12b-position_4_missing'),
    pytest.param('S12b', set_csv_row(asy015(3, 'metadata.csv'), 'LensNA',
                                     '999'),
                 id='S12b-metadata_csv_changed'),
    pytest.param('S12b', left_over(asy015(1, 'GFP.tif')),
                 id='S12b-existing_position_still_written'),
    pytest.param('S12b', replace_last_log_line, id='S12b-log_changed'),
    pytest.param('S09', copy_file(asy015(1, 'GFP.tif'),
                                  asy015(1, 'mNeon_Green.tif')),
                 id='S09-mNeon_Green_holds_GFP'),
    pytest.param('S09', corrupt_pixels(asy015(1, 'GFP.tif')),
                 id='S09-file_main_wrote_changed'),
    pytest.param('S09', set_csv_row(asy015(2, 'metadata.csv'), 'SizeZ', '1'),
                 id='S09-new_metadata_csv_changed'),
    pytest.param('S09', replace_last_log_line, id='S09-log_changed'),
    pytest.param('S07', copy_file(scene(1, 'GFP.tif'), scene(3, 'GFP.tif')),
                 id='S07-position_3_holds_scene_1'),
    pytest.param('S07', drop(scene(3, 'metadata.csv')),
                 id='S07-renamed_file_missing'),
    pytest.param('S07b', copy_file(scene(1, 'GFP.tif'), scene(3, 'GFP.tif')),
                 id='S07b-position_3_holds_scene_1'),
    pytest.param('S07b', set_csv_row(scene(1, 'metadata.csv'), 'SizeT', '2'),
                 id='S07b-metadata_csv_changed'),
    pytest.param('S02', copy_file(pos(2, 'Channel_0_0.tif'),
                                  pos(3, 'Channel_0_0.tif')),
                 id='S02-position_3_holds_file_2'),
    pytest.param('S02', drop(pos(2, 'Channel_0_2.tif')),
                 id='S02-third_channel_missing'),
    pytest.param('S02', replace_last_log_line, id='S02-log_changed'),
    pytest.param('S02', drop_last_dialog, id='S02-dialog_of_file_3_missing'),
])
def test_registry_claims_are_exact(scenario_record, scenario_id, mutate):
    """The registry claims exactly the expected diffs: a scenario whose
    registered output is wrong, missing or left over fails."""
    golden = scenarios.load_golden('module0', scenario_id)
    record = scenario_record(scenario_id)
    changes = expected_module0_changes.CHANGES
    scenarios.assert_golden(scenario_id, golden, record, changes)
    mutate(record, golden)
    with pytest.raises(AssertionError, match='golden mismatch'):
        scenarios.assert_golden(scenario_id, golden, record, changes)
