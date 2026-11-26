"""`convert()` through `SubprocessBackend` (real worker children with the
`FakeBioImage` hook).

- Isolation transparency and `--server` neutrality: a layout-0 and a
  layout-2 conversion through children give the same manifest, log lines and
  result as in-process; both run at the same time in two threads, which also
  shows that concurrent conversions do not mix.
- `test_parent_never_opens_raw`: with `ImageReader`, `bioio.BioImage`
  and the fake reader refusing to open anything in the parent, a full
  interactive conversion (incl. the sample-data read of the dialog) and an
  upfront one succeed.
- Error mapping of the child failures (`MetadataReadError` with the failing
  file, `WorkerError` for a job that does not fit its file, `SettingsError`
  from the `check_positions` child before any raw file is moved).
"""
from __future__ import annotations

import threading

import pytest

from cellacdc.acdc_bioio_bioformats import reader
from cellacdc.acdc_datastruct import (
    backends, convert, errors, planning, runner
)

from . import conditions, fake_bioio, manifest
from .test_convert_inprocess import (
    LAYOUT0, LAYOUT1, LAYOUT2, Recorder, ScriptedDialog, fake,
    layout0_inputs, layout1_inputs, layout2_inputs, make_request, overrides
)

pytestmark = conditions.subprocess_heavy


@pytest.fixture
def hooks(monkeypatch):
    """Children honour the reader factory only with the test-hook env."""
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')


def subprocess_backend(tmp_path) -> backends.SubprocessBackend:
    return backends.SubprocessBackend(
        reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH,
        options=runner.RunnerOptions(temp_root=str(tmp_path / 'runs'))
    )


def run_in_threads(jobs: dict) -> dict:
    """Run every `jobs[name]()` in its own thread; returns name -> (value,
    exception)."""
    outcomes = {}

    def target(name, job):
        try:
            outcomes[name] = (job(), None)
        except BaseException as err:
            outcomes[name] = (None, err)

    threads = [
        threading.Thread(target=target, args=item) for item in jobs.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(300)
    return outcomes


def snapshot(src, dst, result, recorder) -> dict:
    """Manifest of dst, log lines and result summary with the destination
    replaced by a placeholder."""
    placeholders = {'<SRC>': str(src), '<DST>': str(dst)}

    def normalise(value):
        if isinstance(value, str):
            return manifest.normalize_path_text(value, placeholders)
        if isinstance(value, dict):
            return {key: normalise(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalise(item) for item in value]
        return value

    return {
        'dst': manifest.build_manifest(
            str(dst), placeholders=placeholders, include_env=False
        ),
        'logs': normalise(recorder.logs),
        'result': normalise(result.to_dict()),
        'progress': (recorder.inits, recorder.steps),
    }


CASES = {
    'layout0': (
        LAYOUT0, layout0_inputs, {},
        dict(positions=frozenset({1, 3}), time_range=(2, 3), to_h5=True),
    ),
    'layout2': (
        LAYOUT2, layout2_inputs, {'other_files': 'copy'}, {},
    ),
}


def test_subprocess_matches_in_process(tmp_path, hooks):
    """Same output through children as in-process, for two conversions run
    concurrently."""
    def conversion(name, backend, out):
        layout, make_inputs, kwargs, flags = CASES[name]
        src = tmp_path / name / 'src'
        dst = tmp_path / name / out
        files = make_inputs(src)
        if name == 'layout2':
            (src / 'ASY015_1_notes.txt').write_text('notes')
        req = make_request(src, dst, layout, files, **kwargs)
        recorder = Recorder(overrides(**flags))
        result = convert.convert(
            req, callbacks=recorder.callbacks, backend=backend
        )
        return snapshot(src, dst, result, recorder)

    expected = {
        name: conversion(
            name, backends.InProcessBackend(
                reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH
            ), 'dst_in_process'
        )
        for name in CASES
    }
    outcomes = run_in_threads({
        name: lambda name=name: conversion(
            name, subprocess_backend(tmp_path / name), 'dst_children'
        )
        for name in CASES
    })
    for name in CASES:
        actual, error = outcomes[name]
        assert error is None, repr(error)
        diffs = manifest.diff_manifests(expected[name]['dst'], actual['dst'])
        assert diffs == [], manifest.format_diffs(diffs)
        assert actual == expected[name]
    assert expected['layout0']['progress'] == ([6], [2, 2, 2])


def test_parent_never_opens_raw(tmp_path, monkeypatch, hooks):
    """The GUI/CLI process never opens a raw file: every open in the
    parent raises, yet both schedules convert through the children."""
    def refuse(*args, **kwargs):
        raise AssertionError('the parent process opened a raw file')

    monkeypatch.setattr(reader.ImageReader, '__init__', refuse)
    monkeypatch.setattr(fake_bioio.FakeBioImage, '__init__', refuse)
    try:
        import bioio
    except ImportError:
        pass
    else:
        monkeypatch.setattr(bioio.BioImage, '__init__', refuse)

    dialog = ScriptedDialog(('ok', {}), read_sample=True)

    def interactive():
        src, dst = tmp_path / 'gui' / 'src', tmp_path / 'gui' / 'dst'
        files = layout1_inputs(src, count=1)
        req = make_request(src, dst, LAYOUT1, files, schedule='interactive',
                           on_existing=planning.OnExisting.ADD)
        return convert.convert(
            req, callbacks=Recorder(dialog).callbacks,
            backend=subprocess_backend(tmp_path / 'gui')
        )

    def upfront():
        src, dst = tmp_path / 'cli' / 'src', tmp_path / 'cli' / 'dst'
        files = layout1_inputs(src, count=2)
        return convert.convert(
            make_request(src, dst, LAYOUT1, files),
            callbacks=Recorder().callbacks,
            backend=subprocess_backend(tmp_path / 'cli')
        )

    outcomes = run_in_threads({'gui': interactive, 'cli': upfront})
    for name, (result, error) in outcomes.items():
        assert error is None, f'{name}: {error!r}'
        assert result.ok
    assert len(outcomes['gui'][0].written) == 1
    assert len(outcomes['cli'][0].written) == 2
    sample = dialog.samples[0]
    assert [array.shape for array in sample] == [(3, 2, 16, 20)] * 2


def test_child_errors_are_mapped(tmp_path, hooks):
    src = tmp_path / 'src'
    good = str(src / fake(src, 'good.czi', T=3))
    bad = str(src / fake(src, 'bad, µ.czi', fail={'stage': 'open'}))
    backend = subprocess_backend(tmp_path)
    with pytest.raises(errors.MetadataReadError) as info:
        backend.read_metadata([good, bad])
    err = info.value
    assert err.filename == 'bad, µ.czi'
    assert 'injected open failure' in err.details
    assert isinstance(err.__cause__, runner.WorkerError)

    # A plan that no longer fits its file (planned for 3 frames, the file
    # now has 2): the child's ValueError arrives as a WorkerError and
    # nothing is written.
    plan = convert.convert(
        make_request(src, tmp_path / 'dst', LAYOUT1, ['good.czi'],
                     dry_run=True),
        callbacks=Recorder().callbacks,
        backend=backends.InProcessBackend(
            reader_factory=fake_bioio.FakeBioImage
        )
    )
    fake(src, 'good.czi', T=2)
    events = []
    with pytest.raises(runner.WorkerError) as info:
        backend.save_positions(
            plan.jobs, {'xml:0': ''}, on_event=events.append
        )
    assert info.value.error_type == 'ValueError'
    assert 'does not fit the 2 frame' in info.value.message
    assert events == [] and not (tmp_path / 'dst').exists()


def test_raw_files_are_checked_in_a_child_before_moving(tmp_path, hooks):
    """Upfront schedule through children: the check_positions task finds
    the file that does not fit (2 frames, its plan 3) before any raw file
    is moved or DST created; the child's ValueError becomes a
    SettingsError naming the file in SRC."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = layout2_inputs(src, overrides={(2, 'GFP'): {'T': 2}})
    before = conditions.tree_snapshot(str(src))
    with pytest.raises(errors.SettingsError) as info:
        convert.convert(
            make_request(src, dst, LAYOUT2, files, move_raw=True),
            callbacks=Recorder().callbacks,
            backend=subprocess_backend(tmp_path)
        )
    assert info.value.message == (
        'Time range 1-3 (1-based, inclusive) does not fit the 2 frame(s) '
        f'of "{src / "ASY015_2_GFP.czi"}" (series 0).'
    )
    cause = info.value.__cause__
    assert type(cause) is ValueError
    assert isinstance(cause.__cause__, runner.WorkerError)
    assert cause.__cause__.task == 'check_positions'
    assert conditions.tree_snapshot(str(src)) == before
    assert not dst.exists()
