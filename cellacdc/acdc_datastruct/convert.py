"""Orchestrator of a Module 0 conversion ("Create data structure from
microscopy files"), shared by the GUI worker and ``acdc-data convert``.

``convert(request, callbacks=..., backend=...)`` reads the metadata of each
metadata unit, asks ``callbacks.confirm`` for the settings, plans the
Positions and lets the backend write them. Two schedules call the same core
functions:

- ``'interactive'`` (GUI): the raw files are moved first, then per file:
  read the metadata, confirm (the metadata dialog; "Trust" and "Use the
  above metadata" skip it for the next files and re-open it for a file that
  does not fit), write.
- ``'upfront'`` (command line): read every file's metadata in one task,
  confirm (never prompts) and plan everything, check the destination for
  existing Positions, check every planned Position against all its raw
  files and scenes in a second task, then (unless a dry run) move/rename
  the raw files and write everything in a third task. Every ``UsageError``
  is raised before the source or destination folder is changed.

The parent process never opens a raw file: the backend does (in a worker
child in production).
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Mapping

from . import errors, metadata, planning, sources, writers
from .callbacks import ProgressCallbacks

if TYPE_CHECKING:
    from .backends import ConvertBackend

logger = logging.getLogger(__name__)

Schedule = Literal['interactive', 'upfront']
SCHEDULES = ('interactive', 'upfront')
READING_METADATA_TEXT = 'Reading OME metadata...'
_RAW_FILES_HINT = (
    'The plan comes from the metadata of the first file (one-per-channel) '
    'or of the first scene (single-multi-pos), and every file and scene '
    'must fit it. Choose frames that every file and scene has with '
    '--time-range, or leave the file or scene out with --positions or '
    '--save-channels.'
)


@dataclass(frozen=True)
class ConvertRequest:
    """What to convert and how (absolute paths only)."""
    src_dir: str
    dst_dir: str
    layout: sources.RawLayout
    files: tuple[str, ...]                  # ORIGINAL names (= input order)
    on_existing: planning.OnExisting = planning.OnExisting.ERROR
    start_pos_n: int | None = None          # None: from on_existing
    move_raw: bool = False
    renames: Mapping[str, str] | None = None    # --rename-invalid
    use_symlinks: bool = False
    lazy_load: bool = True
    other_files: Literal['leave', 'move', 'copy'] = 'leave'  # layout 2 only
    one_per_channel: sources.OnePerChannelIndex | None = None
    schedule: Schedule = 'upfront'
    dry_run: bool = False                   # upfront only


@dataclass(frozen=True)
class ConfirmContext:
    """Everything the ``confirm`` hook needs to settle one metadata unit."""
    unit: sources.MetadataUnit
    layout: sources.RawLayout
    filename: str                   # == unit.metadata_file
    raw_path: str                   # absolute path the metadata was read from
    raw: metadata.RawFileMetadata
    resolved: metadata.ResolvedMetadata
    warnings: tuple[metadata.MetadataWarning, ...]
    previous: metadata.ConversionSettings | None
    read_sample: Callable[[int, int, int], list]
    """``read_sample(size_c, size_t, size_z)``: sample arrays of
    ``raw_path`` (one backend task per call)."""


@dataclass(frozen=True)
class ConfirmResult:
    """Settings confirmed for one metadata unit and what to do with the next
    units ('ask', 'trust' the reader, 'reuse' these settings)."""
    settings: metadata.ConversionSettings
    apply_to_next: metadata.ApplyToNext = 'ask'


@dataclass
class ConvertCallbacks(ProgressCallbacks):
    """Progress callbacks plus the hooks of a conversion.

    Attributes
    ----------
    confirm : callable
        ``confirm(ConfirmContext) -> ConfirmResult | None`` (required). The
        GUI shows the metadata dialog (None = Cancel), the command line
        derives the settings from the file and its options.
    warn : callable, optional
        ``warn(MetadataWarning)``; default: ``log(line, 'WARNING')`` for each
        line of ``warning.gui_lines()``.
    """
    confirm: Callable[[ConfirmContext], ConfirmResult | None] | None = None
    warn: Callable[[metadata.MetadataWarning], None] | None = None


@dataclass(frozen=True)
class WrittenPosition:
    """A Position written completely (from a 'position_done' event)."""
    input_id: int
    output_number: int
    images_path: str
    files: tuple[str, ...]
    source_file: str


@dataclass
class ConvertResult:
    """Outcome of ``convert``.

    ``written`` is built from the backend's 'position_done' events only.
    ``moved``/``renamed`` are the raw-file moves and renames done, each
    recorded once it is done (dry run: planned); ``deletions`` the existing
    ``Images`` folders that OVERWRITE removed, from the backend's
    'images_deleted' events (dry run: would remove; after a failure also a
    folder whose deletion stopped part way). After a failure they list
    exactly what was changed.
    """
    jobs: list[planning.PositionJob] = field(default_factory=list)
    skipped: list[planning.SkippedPosition] = field(default_factory=list)
    written: list[WrittenPosition] = field(default_factory=list)
    moved: list[tuple[str, str]] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[metadata.MetadataWarning] = field(default_factory=list)
    deletions: list[str] = field(default_factory=list)
    cancelled: bool = False
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        """Not cancelled and (a dry run or at least one Position written).
        The GUI ignores the "at least one written" part."""
        return not self.cancelled and (self.dry_run or len(self.written) > 0)

    def to_dict(self) -> dict:
        """JSON-able summary (command line output): ``positions`` (planned
        for a dry run, else written; each with input_id, output, source
        file, series, Images folder, saved channels, frames, files and the
        other files moved or copied into it), ``skipped`` input ids,
        ``moved``, ``renamed``, ``deleted_images_folders``, ``warnings``,
        ``cancelled``, ``dry_run``."""
        jobs = {(job.input_id, job.output_number): job for job in self.jobs}
        if self.dry_run:
            positions = [_position_dict(job) for job in self.jobs]
        else:
            positions = [
                _position_dict(
                    jobs.get((done.input_id, done.output_number)), done
                )
                for done in self.written
            ]
        return {
            'dry_run': self.dry_run,
            'cancelled': self.cancelled,
            'positions': positions,
            'skipped': [position.input_id for position in self.skipped],
            'moved': [list(pair) for pair in self.moved],
            'renamed': [list(pair) for pair in self.renamed],
            'deleted_images_folders': list(self.deletions),
            'warnings': [
                {'field': w.field, 'file': w.path, 'message': w.message}
                for w in self.warnings
            ],
        }


def _position_dict(
        job: planning.PositionJob | None,
        written: WrittenPosition | None = None
    ) -> dict:
    """One entry of ``ConvertResult.to_dict()['positions']``.

    ``other_files``: the other-file moves/copies of the job (one file per
    channel), all of them for a planned Position, the ones done (their
    target is among the written files) for a written one.
    """
    other_files = () if job is None else job.other_files
    if written is None:
        input_id, output_number = job.input_id, job.output_number
        images_path, files, source_file = job.images_path, (), job.source_file
    else:
        input_id, output_number = written.input_id, written.output_number
        images_path, files = written.images_path, written.files
        source_file = written.source_file
        other_files = [
            action for action in other_files if action.dst in files
        ]
    return {
        'input_id': input_id,
        'output': f'Position_{output_number}',
        'source_file': source_file,
        'series': None if job is None else job.series,
        'images_path': images_path,
        'channels': (
            [] if job is None else [channel.name for channel in job.channels]
        ),
        'time_range': None if job is None else list(job.time_range),
        'files': list(files),
        'other_files': [
            {'src': action.src, 'dst': action.dst, 'mode': action.mode}
            for action in other_files
        ],
    }


def partial_result(err: BaseException) -> ConvertResult | None:
    """The ``ConvertResult`` built until ``convert`` raised ``err`` (what was
    renamed, moved, deleted and written before the failure), or None."""
    return getattr(err, 'convert_result', None)


# ------------------------------------------------------------- preparation
def _check_destination(dst_dir: str) -> None:
    """``dst_dir`` is a folder or can be created in a writable folder."""
    if os.path.lexists(dst_dir) and not os.path.isdir(dst_dir):
        raise errors.PlanError(
            f'The destination is not a folder: {dst_dir}',
            hint='Choose another destination folder.'
        )
    ancestor = dst_dir
    while not os.path.lexists(ancestor):
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            break
        ancestor = parent
    if not os.path.isdir(ancestor):
        raise errors.PlanError(
            f'The destination folder {dst_dir} cannot be created: {ancestor} '
            'is not a folder.', hint='Choose another destination folder.'
        )
    if not os.access(ancestor, os.W_OK):
        raise errors.PlanError(
            f'The destination folder {dst_dir} cannot be written: '
            f'{ancestor} is not writable.',
            hint='Choose another destination folder or check its permissions.'
        )


def prepare(request: ConvertRequest) -> planning.PlanContext:
    """Validate ``request`` and build the planning context. Read-only and
    idempotent (the command line calls it for early validation, ``convert``
    calls it again).

    Checks: absolute source/destination paths, schedule (a dry run needs
    'upfront'), the files exist in the source folder, single-multi-pos has
    exactly one file, one-file-per-channel names parse, start number >= 1,
    the destination is (or can be created as) a writable folder, and the
    planned raw-file renames/moves do not collide.

    Raises
    ------
    UsageError
        Or one of its subclasses (``SourceError``, ``FilenamePatternError``,
        ``PlanError``); nothing else.
    """
    src_dir, dst_dir = request.src_dir, request.dst_dir
    for name, path in (('source', src_dir), ('destination', dst_dir)):
        if not os.path.isabs(path):
            raise errors.UsageError(
                f'The {name} folder must be an absolute path, got "{path}".'
            )
    if request.schedule not in SCHEDULES:
        raise errors.UsageError(
            f'Unknown schedule {request.schedule!r} (expected one of '
            f'{SCHEDULES}).'
        )
    if request.dry_run and request.schedule != 'upfront':
        raise errors.UsageError('A dry run needs the upfront schedule.')
    layout = sources.RawLayout(request.layout)
    files = tuple(request.files)
    if not os.path.isdir(src_dir):
        raise errors.SourceError(
            f'The source folder does not exist: {src_dir}'
        )
    missing = [
        filename for filename in files
        if not os.path.isfile(os.path.join(src_dir, filename))
    ]
    if missing:
        listing = '\n'.join(f'  - {filename}' for filename in missing)
        raise errors.SourceError(
            f'These files were not found in the source folder {src_dir}:\n'
            f'{listing}'
        )
    one_per_channel = request.one_per_channel
    if (
        layout == sources.RawLayout.ONE_FILE_PER_CHANNEL
        and one_per_channel is None
    ):
        one_per_channel = sources.parse_one_per_channel_filenames(files)
    sources.metadata_units(layout, files, one_per_channel)

    on_existing = planning.OnExisting(request.on_existing)
    _check_destination(dst_dir)
    existing = planning.existing_position_numbers(dst_dir)
    start_pos_n = request.start_pos_n
    if start_pos_n is None:
        start_pos_n = planning.start_position_number(on_existing, existing)
    if start_pos_n < 1:
        raise errors.UsageError(
            f'The first Position number must be at least 1, got '
            f'{start_pos_n}.'
        )
    mutations = planning.plan_source_mutations(
        src_dir, files, dst_dir=dst_dir, move_raw=request.move_raw,
        renames=request.renames
    )
    return planning.PlanContext(
        layout=layout, src_dir=src_dir, dst_dir=dst_dir, files=files,
        on_existing=on_existing, existing=tuple(existing),
        start_pos_n=start_pos_n, use_symlinks=request.use_symlinks,
        lazy_load=request.lazy_load, other_files=request.other_files,
        one_per_channel=one_per_channel, mutations=mutations,
        source_listing=tuple(sources.list_source_files(src_dir)),
    )


def read_file_metadata(
        path: str, *, backend: ConvertBackend
    ) -> tuple[
        metadata.RawFileMetadata, metadata.ResolvedMetadata,
        tuple[metadata.MetadataWarning, ...]
    ]:
    """Raw and resolved metadata (one-file-per-position semantics) of one
    raw file and its "not found" warnings (``acdc-data metadata``).

    Raises
    ------
    UsageError
        ``path`` is not absolute.
    MetadataReadError
        The file could not be read.
    """
    if not os.path.isabs(path):
        raise errors.UsageError(
            f'The raw file must be an absolute path, got "{path}".'
        )
    raw = backend.read_metadata([path])[0]
    resolved, warnings = metadata.resolve_metadata(
        raw, layout=sources.RawLayout.ONE_FILE_PER_POS
    )
    return raw, resolved, warnings


def confirm_from_metadata(
        overrides: metadata.MetadataOverrides | None = None, *,
        on_warning: Callable[[str], None] | None = None
    ) -> Callable[[ConfirmContext], ConfirmResult]:
    """The command line ``confirm`` hook: trust the file metadata and apply
    ``overrides`` (``metadata.settings_from_metadata``); never prompts.

    The warnings of ``settings_from_metadata`` (e.g. a channel without a
    name) go to ``on_warning``. A ``SettingsError`` propagates (before any
    file-system change in the upfront schedule). A one-per-pos file that
    ``overrides.positions`` does not select gets placeholder settings
    (``_unselected_file_settings``) without warnings: nothing of it is
    written, so neither the overrides nor its own metadata can reject the
    run.
    """
    def confirm(ctx: ConfirmContext) -> ConfirmResult:
        if (
            overrides is not None and overrides.positions is not None
            and ctx.layout == sources.RawLayout.ONE_FILE_PER_POS
            and ctx.unit.file_ordinal not in overrides.positions
        ):
            return ConfirmResult(
                _unselected_file_settings(ctx.resolved, overrides)
            )
        settings, warnings = metadata.settings_from_metadata(
            ctx.resolved, overrides, layout=ctx.layout,
            context=ctx.unit.label, scene_names=ctx.raw.scene_names
        )
        if on_warning is not None:
            for text in warnings:
                on_warning(text)
        return ConfirmResult(settings)
    return confirm


def _unselected_file_settings(
        resolved: metadata.ResolvedMetadata,
        overrides: metadata.MetadataOverrides) -> metadata.ConversionSettings:
    """Settings of a one-per-pos file that ``overrides.positions`` does not
    select. Planning uses only the selection, SizeC (the progress steps of
    the skipped Position) and SizeT (its default time range), so the other
    values are placeholders that always pass ``validate``: channels named
    'channel_<c>' and all saved, emission wavelengths 0.0, time increment
    and pixel sizes 1.0. No other override and no file value that needs
    checking or converting is used, so overrides that do not fit the file
    and invalid file values (duplicate channel names, unknown units) cannot
    reject the run."""
    size_c = resolved.size_c
    return metadata.ConversionSettings(
        to_h5=overrides.to_h5,
        lens_na=resolved.lens_na,
        size_t=resolved.size_t,
        size_z=resolved.size_z,
        size_c=size_c,
        size_s=resolved.size_s,
        time_increment_s=1.0,
        physical_size_x_um=1.0,
        physical_size_y_um=1.0,
        physical_size_z_um=1.0,
        channel_names=tuple(f'channel_{c}' for c in range(size_c)),
        save_channels=(True,) * size_c,
        em_wavelens=(0.0,) * size_c,
        selected_positions=overrides.positions,
    )


# --------------------------------------------------------------- conversion
def convert(
        request: ConvertRequest, *, callbacks: ConvertCallbacks,
        backend: ConvertBackend
    ) -> ConvertResult:
    """Run the conversion described by ``request`` (see the module
    docstring for the two schedules).

    Progress: ``progress_init`` once (the GUI total of the first unit,
    ``num_positions * SizeC``, for 'interactive'; the sum of every
    position's SizeC for 'upfront'), then one ``progress_step(SizeC)`` per
    position, saved or skipped. Log lines of the backend go to
    ``callbacks.log``.

    Returns the ``ConvertResult``; ``cancelled`` is True when ``confirm``
    returned None.

    Raises
    ------
    UsageError
        Invalid request, settings or plan (upfront: also a ``SettingsError``
        when a planned Position does not fit one of its raw files or
        scenes); in the upfront schedule always before any file-system
        change.
    MetadataReadError, MoveRawFilesError, ConversionCancelled
        Reading a file failed; moving/renaming the raw files failed;
        ``callbacks.is_cancelled()`` returned True.
    runner.WorkerError
        A worker task failed (any other exception with the in-process
        backend).

    Every exception carries the result built so far (see
    ``partial_result``).
    """
    if callbacks.confirm is None:
        raise ValueError('ConvertCallbacks.confirm is required.')
    result = ConvertResult(dry_run=request.dry_run)
    try:
        conversion = _Conversion(
            request, prepare(request), callbacks, backend, result
        )
        if request.schedule == 'interactive':
            conversion.run_interactive()
        else:
            conversion.run_upfront()
    except BaseException as err:
        try:
            err.convert_result = result
        except (AttributeError, TypeError):
            # Exception types without an instance __dict__.
            pass
        raise
    return result


def _xml_ref(unit: sources.MetadataUnit) -> str:
    return f'xml:{unit.index}'


def _with_raw_paths(
        job: planning.PositionJob, paths: Mapping[str, str]
    ) -> planning.PositionJob:
    """``job`` reading its channels from ``paths[raw_path]`` (when
    present) instead of ``raw_path``."""
    return dataclasses.replace(job, channels=tuple(
        dataclasses.replace(
            channel, raw_path=paths.get(channel.raw_path, channel.raw_path)
        )
        for channel in job.channels
    ))


class _Conversion:
    """State of one ``convert`` call."""
    def __init__(
            self, request: ConvertRequest, ctx: planning.PlanContext,
            callbacks: ConvertCallbacks, backend: ConvertBackend,
            result: ConvertResult
        ) -> None:
        self.request = request
        self.ctx = ctx
        self.callbacks = callbacks
        self.backend = backend
        self.result = result
        self.units = sources.metadata_units(
            ctx.layout, ctx.files, ctx.one_per_channel
        )

    # ---------------------------------------------------------- schedules
    def run_interactive(self) -> None:
        """GUI order: move first, then per unit read, confirm (or Trust /
        Reuse), plan and write."""
        ctx, callbacks = self.ctx, self.callbacks
        if not ctx.mutations.is_empty():
            self._execute_mutations()
        mode: metadata.ApplyToNext = 'ask'
        previous = None
        initialised = False
        for unit in self.units:
            self._check_cancelled()
            callbacks.log(READING_METADATA_TEXT)
            raw_path = ctx.mutations.final_paths[unit.metadata_file]
            raw = self.backend.read_metadata([raw_path])[0]
            resolved, warnings = self._resolve(raw)
            settings = None
            if mode != 'ask':
                settings = metadata.settings_for_next_unit(
                    mode, previous, resolved
                )
            if settings is None:
                answer = callbacks.confirm(self._confirm_context(
                    unit, raw_path, raw, resolved, warnings, previous
                ))
                if answer is None:
                    self.result.cancelled = True
                    return
                settings, mode = answer.settings, answer.apply_to_next
            jobs, skipped = self._plan(unit, raw, settings)
            planning.check_conflicts(jobs, ctx.existing, ctx.on_existing)
            if not initialised:
                num_positions = planning.num_positions(
                    ctx.layout, settings, num_files=len(ctx.files),
                    one_per_channel=ctx.one_per_channel
                )
                callbacks.progress_init(num_positions * settings.size_c)
                initialised = True
            for position in skipped:
                callbacks.progress_step(position.size_c)
            if jobs:
                self._save(jobs, {_xml_ref(unit): raw.metadata_xml})
            previous = settings

    def run_upfront(self) -> None:
        """Command line order: read every unit, confirm and plan all, check
        conflicts and the raw files (dry run: stop here), then move/rename
        and write all."""
        ctx, callbacks, result = self.ctx, self.callbacks, self.result
        self._check_cancelled()
        paths = [
            os.path.join(ctx.src_dir, unit.metadata_file)
            for unit in self.units
        ]
        callbacks.log(f'Reading the metadata of {len(paths)} file(s)...')
        raws = self.backend.read_metadata(paths)
        texts = {}
        previous = None
        for unit, path, raw in zip(self.units, paths, raws, strict=True):
            resolved, warnings = self._resolve(raw)
            answer = callbacks.confirm(self._confirm_context(
                unit, path, raw, resolved, warnings, previous
            ))
            if answer is None:
                result.cancelled = True
                return
            self._plan(unit, raw, answer.settings)
            texts[_xml_ref(unit)] = raw.metadata_xml
            previous = answer.settings
        planning.check_conflicts(result.jobs, ctx.existing, ctx.on_existing)
        if not result.jobs:
            raise errors.SettingsError(
                'No positions selected: there is nothing to convert.',
                hint='Select existing input positions with --positions.'
            )
        self._check_raw_files(result.jobs)
        if self.request.dry_run:
            result.moved = [
                (move.src, move.dst) for move in ctx.mutations.moves
            ]
            result.renamed = list(ctx.mutations.renames)
            return

        self._check_cancelled()
        os.makedirs(ctx.dst_dir, exist_ok=True)
        if not ctx.mutations.is_empty():
            self._execute_mutations()
        callbacks.progress_init(
            sum(len(job.metadata_csv.channel_names) for job in result.jobs)
            + sum(position.size_c for position in result.skipped)
        )
        for position in result.skipped:
            callbacks.progress_step(position.size_c)
        self._save(result.jobs, texts)

    # ------------------------------------------------------------- steps
    def _check_cancelled(self) -> None:
        if self.callbacks.is_cancelled():
            raise errors.ConversionCancelled('The conversion was cancelled.')

    def _check_raw_files(self, jobs: list[planning.PositionJob]) -> None:
        """Upfront schedule: check every job against all its raw files and
        scenes (time range, z-slices, channel indices; one backend task)
        at their current paths, before anything is renamed, moved or
        written.

        Raises
        ------
        SettingsError
            A job does not fit one of its files (the message of the check,
            naming the file and the series).
        """
        self._check_cancelled()
        self.callbacks.log(
            f'Checking the raw files of {len(jobs)} position(s)...'
        )
        current = {
            final: os.path.join(self.ctx.src_dir, filename)
            for filename, final in self.ctx.mutations.final_paths.items()
        }
        try:
            self.backend.check_positions([
                _with_raw_paths(job, current) for job in jobs
            ])
        except errors.UsageError:
            raise
        except ValueError as err:
            raise errors.SettingsError(
                str(err), hint=_RAW_FILES_HINT
            ) from err

    def _execute_mutations(self) -> None:
        """Rename, then move the raw files one file at a time, recording
        each rename and move in the result once it is done, so that after a
        failure or an interruption the result lists exactly what changed."""
        result = self.result
        for pair in self.ctx.mutations.renames:
            self._execute_mutation(renames=(pair,))
            result.renamed.append(pair)
        for move in self.ctx.mutations.moves:
            self._execute_mutation(moves=(move,))
            result.moved.append((move.src, move.dst))

    def _execute_mutation(
            self, *, renames: tuple[tuple[str, str], ...] = (),
            moves: tuple[planning.RawMove, ...] = ()
        ) -> None:
        """One step of ``_execute_mutations``; the ``moved`` of a
        ``MoveRawFilesError`` lists every move of the run done so far."""
        step = planning.SourceMutations(
            renames=renames, moves=moves, final_paths={}
        )
        try:
            writers.execute_source_mutations(step, log=self.callbacks.log)
        except errors.MoveRawFilesError as err:
            err.moved = list(self.result.moved)
            raise

    def _resolve(
            self, raw: metadata.RawFileMetadata
        ) -> tuple[
            metadata.ResolvedMetadata, tuple[metadata.MetadataWarning, ...]
        ]:
        resolved, warnings = metadata.resolve_metadata(
            raw, layout=self.ctx.layout,
            one_per_channel=self.ctx.one_per_channel
        )
        for warning in warnings:
            if self.callbacks.warn is None:
                for line in warning.gui_lines():
                    self.callbacks.log(line, 'WARNING')
            else:
                self.callbacks.warn(warning)
        self.result.warnings.extend(warnings)
        return resolved, warnings

    def _confirm_context(
            self, unit: sources.MetadataUnit, raw_path: str,
            raw: metadata.RawFileMetadata,
            resolved: metadata.ResolvedMetadata,
            warnings: tuple[metadata.MetadataWarning, ...],
            previous: metadata.ConversionSettings | None
        ) -> ConfirmContext:
        def read_sample(size_c: int, size_t: int, size_z: int) -> list:
            return self.backend.read_sample(
                raw_path, size_c=size_c, size_t=size_t, size_z=size_z,
                lazy_load=self.ctx.lazy_load
            )

        return ConfirmContext(
            unit=unit, layout=self.ctx.layout, filename=unit.metadata_file,
            raw_path=raw_path, raw=raw, resolved=resolved,
            warnings=warnings, previous=previous, read_sample=read_sample,
        )

    def _plan(
            self, unit: sources.MetadataUnit, raw: metadata.RawFileMetadata,
            settings: metadata.ConversionSettings
        ) -> tuple[list[planning.PositionJob], list[planning.SkippedPosition]]:
        """Validate the settings (backstop of the dialog/resolver), plan the
        unit's Positions and record them in the result (dry run: also the
        ``Images`` folders OVERWRITE would delete)."""
        settings.validate(context=unit.label)
        jobs, skipped = planning.plan_unit_positions(
            self.ctx, unit, raw, settings, xml_ref=_xml_ref(unit)
        )
        self.result.jobs.extend(jobs)
        self.result.skipped.extend(skipped)
        if self.request.dry_run:
            self.result.deletions.extend(
                job.images_path for job in jobs
                if job.clear_images and os.path.isdir(job.images_path)
            )
        return jobs, skipped

    def _save(
            self, jobs: list[planning.PositionJob], texts: Mapping[str, str]
        ) -> None:
        """Write ``jobs`` with one backend task; 'position_done' events are
        the only source of ``result.written``, 'images_deleted' events of
        ``result.deletions``."""
        by_key = {(job.input_id, job.output_number): job for job in jobs}
        from_events: list[WrittenPosition] = []

        def on_event(event: dict) -> None:
            kind = event.get('kind')
            if kind == 'log':
                self.callbacks.log(
                    event.get('message', ''), event.get('level', 'INFO')
                )
            elif kind == 'position_done':
                job = by_key[(event['input_id'], event['output_number'])]
                position = WrittenPosition(
                    input_id=event['input_id'],
                    output_number=event['output_number'],
                    images_path=event['images_path'],
                    files=tuple(event['files']),
                    source_file=job.source_file,
                )
                from_events.append(position)
                self.result.written.append(position)
                self.callbacks.progress_step(
                    len(job.metadata_csv.channel_names)
                )
            elif kind == 'images_deleted':
                self.result.deletions.append(event['images_path'])
            else:
                logger.debug('Worker event: %r', event)

        returned = self.backend.save_positions(jobs, texts, on_event=on_event)
        reported = [
            (entry['input_id'], entry['output_number'], entry['images_path'],
             tuple(entry['files']))
            for entry in returned
        ]
        seen = [
            (position.input_id, position.output_number,
             position.images_path, position.files)
            for position in from_events
        ]
        if reported != seen:
            self.callbacks.log(
                'The written Positions reported by the worker do not match '
                f'its events (reported {len(reported)}, events '
                f'{len(seen)}).', 'WARNING'
            )
