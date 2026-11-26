"""Tasks that open raw microscopy files: they run in a worker child
(``python -m cellacdc.acdc_datastruct._worker``) or, in tests and for the
golden comparison, in-process (``backends.InProcessBackend``).

Every task is ``function(params: dict, ctx: TaskContext) -> dict`` with
JSON-serialisable params, events and results (the subprocess transport is
JSON). ``TASKS`` maps the task names of the runner protocol to them.

Events passed to ``ctx.emit``:

- ``{'kind': 'log', 'level': 'INFO', 'message': str}``: a line for the user
  (the GUI log, the command line log).
- ``{'kind': 'channel_done', 'name': str}``: informational.
- ``{'kind': 'images_deleted', 'images_path': str}``: an existing Images
  folder was deleted (OVERWRITE), or partly deleted when the deletion
  failed; the parent reports these folders.
- ``{'kind': 'position_done', 'input_id', 'output_number', 'images_path',
  'files'}``: one Position was written completely (the parent builds its
  list of written positions from these events only).

The modules that open raw files (``acdc_bioio_bioformats.raw_metadata``,
``reader`` and ``writer``) are imported inside the tasks, so ``ping`` works
even when a reader dependency is broken.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from . import naming, planning, runner, writers

SAMPLE_MAX_T = 4
SAMPLE_MAX_Z = 20


@dataclass
class TaskContext:
    """What a task gets besides its params.

    Attributes
    ----------
    out_dir : str
        Per-run folder (absolute) where a task may save files for the
        parent (e.g. the sample arrays).
    emit : callable
        Event sink: ``emit(event_dict)`` (child: one event line on the
        event channel; in-process: the caller's callback).
    reader_factory : callable, optional
        Replaces ``bioio.BioImage`` (test hook only, e.g. ``FakeBioImage``).
    """
    out_dir: str
    emit: Callable[[dict], None]
    reader_factory: Callable | None = None


class TaskError(Exception):
    """A task failed; ``context`` (e.g. ``{'path': failing file}``) is
    copied into the worker's ``error.json`` and reaches the parent as
    ``runner.WorkerError.context``."""
    def __init__(
            self, message: str, *,
            context: Mapping[str, object] | None = None
        ) -> None:
        super().__init__(message)
        self.context = dict(context or {})


def _log_event(message: str, level: str = 'INFO') -> dict:
    return {'kind': 'log', 'level': level, 'message': message}


def ping(params: dict, ctx: TaskContext) -> dict:
    """Identity of the process running the tasks: ``cellacdc_file``,
    ``python``, ``pid`` and the runner ``protocol``."""
    import cellacdc
    return {
        'cellacdc_file': cellacdc.__file__,
        'python': sys.executable,
        'pid': os.getpid(),
        'protocol': runner.PROTOCOL,
    }


def init_reader(params: dict, ctx: TaskContext) -> dict:
    """Open and close an ``ImageReader`` on ``params['path']`` (checks that
    the file can be read with the installed reader plugins)."""
    from ..acdc_bioio_bioformats import reader
    path = params['path']
    with reader.ImageReader(
            path, qparent=None, bioimage_factory=ctx.reader_factory):
        pass
    return {'path': path}


def read_metadata(params: dict, ctx: TaskContext) -> dict:
    """Raw metadata of every file of ``params['paths']`` (read one after
    the other), as ``RawFileMetadata`` dicts in the same order:
    ``{'metadata': [dict, ...]}``.

    Raises
    ------
    TaskError
        Reading a file failed; ``context={'path': path}``, chained to the
        original exception.
    """
    from ..acdc_bioio_bioformats import raw_metadata
    found = []
    for path in params['paths']:
        try:
            found.append(raw_metadata.read_raw_metadata(
                path, reader_factory=ctx.reader_factory
            ))
        except Exception as err:
            raise TaskError(
                f'Reading the metadata of "{path}" failed: {err}',
                context={'path': path}
            ) from err
    return {'metadata': found}


def read_sample(params: dict, ctx: TaskContext) -> dict:
    """Sample image data shown by the metadata dialog (port of the legacy
    ``_read_sample_data.py`` child).

    params: ``path``, ``size_c``, ``size_t``, ``size_z``, ``lazy_load``,
    optional ``max_t`` (4), ``max_z`` (20) and ``series`` (0). For each
    channel, the first ``min(size_t, max_t)`` frames and
    ``min(size_z, max_z)`` z-slices are read plane by plane, stacked
    (z inside t) and squeezed, then saved as ``out_dir/sample_<c>.npy``.
    Returns ``{'arrays': [file names relative to out_dir]}``.
    """
    import numpy as np

    from ..acdc_bioio_bioformats import reader
    path = params['path']
    size_c = params['size_c']
    sample_size_t = min(params['size_t'], params.get('max_t', SAMPLE_MAX_T))
    sample_size_z = min(params['size_z'], params.get('max_z', SAMPLE_MAX_Z))
    series = params.get('series', 0)
    if min(size_c, sample_size_t, sample_size_z) < 1:
        raise ValueError(
            f'Cannot read sample data of "{path}" with SizeC={size_c}, '
            f'SizeT={params["size_t"]}, SizeZ={params["size_z"]}.'
        )
    channels_data = []
    with reader.ImageReader(
            path, qparent=None, lazy_load=params['lazy_load'],
            bioimage_factory=ctx.reader_factory) as image_reader:
        for c in range(size_c):
            frames = []
            for t in range(sample_size_t):
                planes = [
                    image_reader.read(
                        c=c, z=z, t=t, rescale=False, series=series
                    )
                    for z in range(sample_size_z)
                ]
                frames.append(
                    np.squeeze(np.array(planes, dtype=planes[-1].dtype))
                )
            channels_data.append(
                np.squeeze(np.array(frames, dtype=frames[-1].dtype))
            )
    filenames = []
    for c, channel_data in enumerate(channels_data):
        filename = f'sample_{c}.npy'
        np.save(os.path.join(ctx.out_dir, filename), channel_data)
        filenames.append(filename)
    return {'arrays': filenames}


def _images_request(job: planning.PositionJob, *, lazy_load: bool):
    from ..acdc_bioio_bioformats import writer
    return writer.PositionImagesRequest(
        series=job.series,
        images_path=job.images_path,
        filename_no_ext=job.filename_no_ext,
        s0p=job.pos_str,
        channels=tuple(
            writer.ChannelImageSpec(
                channel.raw_path, channel.file_channel_index, channel.name
            )
            for channel in job.channels
        ),
        time_range=job.time_range,
        size_z=job.size_z,
        time_increment_s=job.time_increment_s,
        physical_size_zyx_um=job.physical_size_zyx_um,
        fmt=job.output,
        lazy_load=lazy_load,
        image_name=job.image_name,
    )


def _release_readers(reader_cache: dict, raw_paths: set[str]) -> None:
    """Close and forget the cached readers of ``raw_paths`` (the writer
    keys its cache by ``(raw_path, lazy_load)``)."""
    for key in [key for key in reader_cache if key[0] in raw_paths]:
        reader_cache.pop(key).close()


def _last_use(jobs: Sequence[planning.PositionJob]) -> dict[str, int]:
    """raw_path -> index of the last job that reads it."""
    return {
        channel.raw_path: index
        for index, job in enumerate(jobs) for channel in job.channels
    }


def _check_job(
        job: planning.PositionJob, ctx: TaskContext, reader_cache: dict
    ) -> None:
    """``writer.check_position_images`` of ``job`` (files opened lazily,
    readers kept in ``reader_cache``)."""
    from ..acdc_bioio_bioformats import writer
    writer.check_position_images(
        _images_request(job, lazy_load=True),
        reader_factory=ctx.reader_factory, reader_cache=reader_cache
    )


def check_positions(params: dict, ctx: TaskContext) -> dict:
    """Check every job against its raw files without writing anything: the
    checks ``save_positions`` runs first (time range, z-slices, channel
    indices of every raw file and series; ``writer.check_position_images``).
    The upfront schedule runs it before any raw file is renamed or moved.

    params: ``{'jobs': [PositionJob dict, ...]}``.

    The files are opened lazily (only their dimensions are read); each one
    is closed after the last job that uses it, and every reader is closed
    when the task ends (also on errors).

    Returns ``{'checked': <number of jobs>}``.

    Raises
    ------
    ValueError
        A job does not fit one of its raw files (the message names the file
        and the series).
    """
    jobs = [planning.PositionJob.from_dict(job) for job in params['jobs']]
    last_use = _last_use(jobs)
    reader_cache: dict = {}
    try:
        for index, job in enumerate(jobs):
            _check_job(job, ctx, reader_cache)
            _release_readers(reader_cache, {
                path for path, last in last_use.items() if last == index
            })
    finally:
        for image_reader in reader_cache.values():
            image_reader.close()
    return {'checked': len(jobs)}


def _count_entries(path: str) -> int:
    """Files and folders inside ``path`` (recursively)."""
    return sum(len(dirs) + len(files) for _, dirs, files in os.walk(path))


def _prepare_images_folder(
        job: planning.PositionJob, ctx: TaskContext
    ) -> None:
    """``writers.prepare_images_folder`` of ``job``. Deleting an existing
    ``Images`` folder (OVERWRITE) emits 'images_deleted', also when the
    deletion stopped part way (e.g. a file open in another program on
    Windows) or creating the new folder failed, whenever something of the
    old folder is gone."""
    clears = job.clear_images and os.path.exists(job.images_path)
    entries = _count_entries(job.images_path) if clears else 0
    deleted = {'kind': 'images_deleted', 'images_path': job.images_path}
    try:
        writers.prepare_images_folder(
            job.images_path, clear=job.clear_images
        )
    except BaseException:
        if clears and (
            not os.path.lexists(job.images_path)
            or _count_entries(job.images_path) != entries
        ):
            ctx.emit(deleted)
        raise
    if clears:
        ctx.emit(deleted)


def save_positions(params: dict, ctx: TaskContext) -> dict:
    """Write complete Positions in the legacy order.

    params: ``{'jobs': [PositionJob dict, ...], 'texts': {ref: xml}}``.

    First every job is checked against its raw files (time range, z-slices,
    channel indices; ``writer.check_position_images``, as in
    ``check_positions``), so a job that does not fit fails before any
    folder is cleared or created. Then, per job: ``prepare_images_folder``
    (an OVERWRITE job deletes its existing ``Images`` folder: event
    ``{'kind': 'images_deleted', 'images_path'}``, also emitted before the
    error when the deletion failed part way),
    log ``'Position <n>/<num>: saving data to <Images>...'``,
    ``<basename>metadataXML.txt``, ``<basename>metadata.csv``, the image
    files (``writer.save_position_images``), the other files (one file per
    channel), a check that every expected file exists, then the
    ``position_done`` event.

    One reader cache is shared by the checks and the writes of this call;
    each raw file is closed after the last job that uses it, and every
    reader is closed when the task ends (also on errors). The checks open
    the files lazily (only their dimensions are read).

    Returns ``{'written': [{input_id, output_number, images_path, files},
    ...]}`` (the parent uses it only as a cross-check of the events).

    Raises
    ------
    ValueError
        A job does not fit its raw files, or its metadataXML text is
        missing (nothing written).
    TaskError
        An expected output file is missing after writing
        (``context={'path': path}``).
    """
    from ..acdc_bioio_bioformats import writer
    jobs = [planning.PositionJob.from_dict(job) for job in params['jobs']]
    texts = params['texts']
    missing_refs = sorted(
        {job.metadata_xml_ref for job in jobs} - set(texts)
    )
    if missing_refs:
        raise ValueError(f'No metadataXML text for {missing_refs}.')

    def log(message: str, level: str = 'INFO') -> None:
        ctx.emit(_log_event(message, level))

    def channel_done(name: str) -> None:
        ctx.emit({'kind': 'channel_done', 'name': name})

    last_use = _last_use(jobs)
    reader_cache: dict = {}
    written = []
    try:
        for job in jobs:
            _check_job(job, ctx, reader_cache)
        for index, job in enumerate(jobs):
            _prepare_images_folder(job, ctx)
            log(
                f'Position {job.output_number}/{job.num_positions}: '
                f'saving data to {job.images_path}...'
            )
            xml_path = os.path.join(
                job.images_path,
                naming.position_filename(job.basename, 'metadataXML', '.txt')
            )
            writers.write_metadata_xml(
                xml_path, texts[job.metadata_xml_ref]
            )
            csv_path = os.path.join(
                job.images_path,
                naming.position_filename(job.basename, 'metadata', '.csv')
            )
            writers.write_metadata_csv(csv_path, job.metadata_csv)
            image_paths = writer.save_position_images(
                _images_request(job, lazy_load=job.lazy_load),
                reader_factory=ctx.reader_factory, reader_cache=reader_cache,
                on_channel_done=channel_done
            )
            other_paths = writers.execute_other_files(
                job.other_files, log=log
            )
            expected = [xml_path, csv_path, *image_paths]
            for path in expected:
                if not os.path.exists(path):
                    raise TaskError(
                        f'expected output missing: "{path}"',
                        context={'path': path}
                    )
            entry = {
                'input_id': job.input_id,
                'output_number': job.output_number,
                'images_path': job.images_path,
                'files': expected + other_paths,
            }
            ctx.emit({'kind': 'position_done', **entry})
            written.append(entry)
            _release_readers(reader_cache, {
                path for path, last in last_use.items() if last == index
            })
    finally:
        for image_reader in reader_cache.values():
            image_reader.close()
    return {'written': written}


TASKS: dict[str, Callable[[dict, TaskContext], dict]] = {
    'ping': ping,
    'init_reader': init_reader,
    'read_metadata': read_metadata,
    'read_sample': read_sample,
    'check_positions': check_positions,
    'save_positions': save_positions,
}
