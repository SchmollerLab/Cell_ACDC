"""What to write where: existing Position folders, raw-file renames and
moves, one ``PositionJob`` per output Position and conflict checks.

Pure (reads the file system at most to check for existing paths, never
changes it). Every path stored in the dataclasses is absolute. This module
never imports ``writers`` (``writers`` imports ``planning``).
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal, Mapping, Sequence

from .. import myutils
from . import errors, metadata, naming, sources

IMAGES_FOLDERNAME = 'Images'
ON_EXISTING_CHOICES = ('overwrite', 'add', 'create-new')


class OnExisting(str, Enum):
    """What to do when the destination already has Position folders."""
    ERROR = 'error'
    OVERWRITE = 'overwrite'
    ADD = 'add'
    CREATE_NEW = 'create-new'

    @classmethod
    def from_gui_flags(
            cls, *, overwrite: bool, add_files: bool, create_new: bool
        ) -> 'OnExisting':
        """Policy of the setup dialog flags: overwrite -> OVERWRITE,
        create_new -> CREATE_NEW, else ADD (which is also what "Add image
        files", "Cancel" and "no existing Position" meant in the GUI, so
        ``add_files`` does not change the result)."""
        if overwrite:
            return cls.OVERWRITE
        if create_new:
            return cls.CREATE_NEW
        return cls.ADD


def existing_position_numbers(dst_dir: str) -> list[int]:
    """Sorted numbers of the valid Position folders of ``dst_dir``
    (``Position_<n>`` with a non-empty ``Images`` folder, as
    ``myutils.get_pos_foldernames``); [] when ``dst_dir`` does not exist."""
    if not os.path.isdir(dst_dir):
        return []
    return sorted(
        int(name.split('_')[-1])
        for name in myutils.get_pos_foldernames(dst_dir)
    )


def start_position_number(
        on_existing: OnExisting, existing: Sequence[int]
    ) -> int:
    """First output Position number: after the highest existing one for
    CREATE_NEW, else 1."""
    if on_existing == OnExisting.CREATE_NEW:
        return max(existing, default=0) + 1
    return 1


def _require_absolute(path: str, what: str) -> None:
    if not os.path.isabs(path):
        raise ValueError(f'{what} must be an absolute path, got {path!r}')


@dataclass(frozen=True)
class MetadataCsvSpec:
    """Values of one Position's metadata.csv (legacy ``saveData``)."""
    lens_na: float
    saved_size_t: int
    size_z: int
    time_increment_s: float
    physical_size_z_um: float
    physical_size_y_um: float
    physical_size_x_um: float
    basename: str
    channel_names: tuple[str, ...]
    save_channels: tuple[bool, ...]
    em_wavelens: tuple[float, ...]

    def to_dict(self) -> dict:
        """JSON-able dict (inverse of ``from_dict``)."""
        data = dataclasses.asdict(self)
        for key in ('channel_names', 'save_channels', 'em_wavelens'):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, d: Mapping) -> 'MetadataCsvSpec':
        """Spec from ``to_dict()`` output."""
        kwargs = dict(d)
        for key in ('channel_names', 'save_channels', 'em_wavelens'):
            kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def metadata_csv_spec(
        settings: metadata.ConversionSettings, *, basename: str,
        saved_size_t: int
    ) -> MetadataCsvSpec:
    """metadata.csv values of a Position written with ``settings``."""
    return MetadataCsvSpec(
        lens_na=settings.lens_na,
        saved_size_t=saved_size_t,
        size_z=settings.size_z,
        time_increment_s=settings.time_increment_s,
        physical_size_z_um=settings.physical_size_z_um,
        physical_size_y_um=settings.physical_size_y_um,
        physical_size_x_um=settings.physical_size_x_um,
        basename=basename,
        channel_names=tuple(settings.channel_names),
        save_channels=tuple(settings.save_channels),
        em_wavelens=tuple(settings.em_wavelens),
    )


def num_positions(
        layout: sources.RawLayout, settings: metadata.ConversionSettings, *,
        num_files: int, one_per_channel: sources.OnePerChannelIndex | None
    ) -> int:
    """Number of positions of a conversion (padding and progress totals):
    single-multi-pos ``settings.size_s``; one-per-pos ``num_files``;
    one-per-channel the number of position ids."""
    if layout == sources.RawLayout.SINGLE_FILE_MULTI_POS:
        return settings.size_s
    if layout == sources.RawLayout.ONE_FILE_PER_POS:
        return num_files
    return len(one_per_channel.position_ids)


# ------------------------------------------------------ source mutations
@dataclass(frozen=True)
class RawMove:
    """Move of one raw file (and of its '_<stem>_' companion folder, e.g.
    for .vsi files) into ``<dst>/raw_microscopy_files``."""
    src: str
    dst: str
    companion_src: str | None
    companion_dst: str | None

    def __post_init__(self) -> None:
        _require_absolute(self.src, 'RawMove.src')
        _require_absolute(self.dst, 'RawMove.dst')
        for name in ('companion_src', 'companion_dst'):
            path = getattr(self, name)
            if path is not None:
                _require_absolute(path, f'RawMove.{name}')


@dataclass(frozen=True)
class SourceMutations:
    """Planned changes of the source files, executed right before
    writing."""
    renames: tuple[tuple[str, str], ...]    # (old abs, new abs)
    moves: tuple[RawMove, ...]
    final_paths: Mapping[str, str]          # ORIGINAL filename -> final path

    def __post_init__(self) -> None:
        for old, new in self.renames:
            _require_absolute(old, 'SourceMutations rename source')
            _require_absolute(new, 'SourceMutations rename target')
        for filename, path in self.final_paths.items():
            _require_absolute(path, f'final path of {filename}')

    def is_empty(self) -> bool:
        """True when nothing is renamed or moved."""
        return not self.renames and not self.moves


def plan_source_mutations(
        src_dir: str, files: Sequence[str], *, dst_dir: str, move_raw: bool,
        renames: Mapping[str, str] | None = None
    ) -> SourceMutations:
    """Plan the renames (command line ``--rename-invalid``) and the moves
    into ``<dst_dir>/raw_microscopy_files`` (after renaming). As in the GUI,
    nothing is moved when ``src_dir`` is itself named
    'raw_microscopy_files'.

    Raises
    ------
    PlanError
        A rename or move target (file or companion folder) already exists,
        two files would get the same name (ignoring case, see
        ``sources.name_collisions``), or a file to rename has a '_<stem>_'
        companion folder (renaming would detach it).
    ValueError
        ``src_dir`` or ``dst_dir`` is not absolute.
    """
    _require_absolute(src_dir, 'src_dir')
    _require_absolute(dst_dir, 'dst_dir')
    renames = {} if renames is None else renames
    problems = []
    rename_pairs = []
    final_names: dict[str, str] = {}
    for filename in files:
        new_name = renames.get(filename, filename)
        final_names[filename] = new_name
        if new_name == filename:
            continue
        old_path = os.path.join(src_dir, filename)
        new_path = os.path.join(src_dir, new_name)
        if os.path.lexists(new_path):
            problems.append(f'cannot rename {filename}: {new_path} exists')
        companion = _companion_path(src_dir, filename)
        if os.path.exists(companion):
            problems.append(
                f'cannot rename {filename}: renaming would detach its '
                f'companion folder {companion}'
            )
        rename_pairs.append((old_path, new_path))

    src_name = os.path.basename(os.path.normpath(src_dir))
    do_move = (
        move_raw and src_name != sources.RAW_MICROSCOPY_FILES_FOLDERNAME
    )
    raw_dir = os.path.join(dst_dir, sources.RAW_MICROSCOPY_FILES_FOLDERNAME)
    problems.extend(sources.name_collisions(
        files, final_names, moved_to=raw_dir if do_move else None
    ))
    moves = []
    final_paths = {}
    for filename in files:
        name = final_names[filename]
        src = os.path.join(src_dir, name)
        if not do_move:
            final_paths[filename] = src
            continue
        dst = os.path.join(raw_dir, name)
        if os.path.lexists(dst):
            problems.append(f'cannot move {name}: {dst} exists')
        companion_src = _companion_path(src_dir, name)
        companion_dst = None
        if os.path.exists(companion_src):
            companion_dst = _companion_path(raw_dir, name)
            if os.path.lexists(companion_dst):
                problems.append(
                    f'cannot move the companion folder of {name}: '
                    f'{companion_dst} exists'
                )
        else:
            companion_src = None
        moves.append(RawMove(src, dst, companion_src, companion_dst))
        final_paths[filename] = dst

    if problems:
        details = '\n'.join(f'  - {problem}' for problem in problems)
        raise errors.PlanError(
            f'The raw files cannot be renamed or moved:\n{details}',
            hint='Move or rename the existing files first.'
        )
    return SourceMutations(
        renames=tuple(rename_pairs), moves=tuple(moves),
        final_paths=final_paths,
    )


def _companion_path(folder: str, filename: str) -> str:
    """'_<stem>_' folder that ``io.move_raw_microscopy_file`` moves together
    with ``filename``."""
    return os.path.join(folder, f'_{os.path.splitext(filename)[0]}_')


# ------------------------------------------------------------ position jobs
@dataclass(frozen=True)
class ChannelSource:
    """One saved channel of a Position."""
    name: str                   # sanitized output name
    csv_index: int              # c of the channel_{c}_* rows (= dialog row)
    raw_path: str               # FINAL absolute raw path
    file_channel_index: int     # channel index inside raw_path

    def __post_init__(self) -> None:
        _require_absolute(self.raw_path, 'ChannelSource.raw_path')


@dataclass(frozen=True)
class OtherFileAction:
    """One-file-per-channel: a non-raw file of a position moved or copied
    into its Images folder."""
    src: str
    dst: str
    mode: Literal['move', 'copy']

    def __post_init__(self) -> None:
        _require_absolute(self.src, 'OtherFileAction.src')
        _require_absolute(self.dst, 'OtherFileAction.dst')


@dataclass(frozen=True)
class PositionJob:
    """Everything needed to write one output Position (JSON via
    ``to_dict``/``from_dict``)."""
    unit_index: int
    input_id: int
    output_number: int
    num_positions: int
    pos_str: str
    images_path: str            # <dst>/Position_<n>/Images
    clear_images: bool          # OnExisting.OVERWRITE
    basename: str
    filename_no_ext: str        # legacy filenameNOext of the pixel writer
    series: int
    source_file: str            # ORIGINAL filename (reports)
    channels: tuple[ChannelSource, ...]     # saved channels, dialog order
    metadata_csv: MetadataCsvSpec
    metadata_xml_ref: str       # key into the job's texts
    time_range: tuple[int, int]             # 1-based inclusive
    size_z: int | None          # None: the writer reads it per raw file
    output: Literal['tif', 'h5', 'symlink']
    lazy_load: bool
    time_increment_s: float
    physical_size_zyx_um: tuple[float, float, float]
    image_name: str = ''
    other_files: tuple[OtherFileAction, ...] = ()

    def __post_init__(self) -> None:
        _require_absolute(self.images_path, 'PositionJob.images_path')
        for channel in self.channels:
            _require_absolute(channel.raw_path, 'PositionJob channel path')
        for action in self.other_files:
            _require_absolute(action.src, 'PositionJob other file source')
            _require_absolute(action.dst, 'PositionJob other file target')

    def to_dict(self) -> dict:
        """JSON-able dict (inverse of ``from_dict``)."""
        data = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }
        data['channels'] = [
            dataclasses.asdict(channel) for channel in self.channels
        ]
        data['metadata_csv'] = self.metadata_csv.to_dict()
        data['time_range'] = list(self.time_range)
        data['physical_size_zyx_um'] = list(self.physical_size_zyx_um)
        data['other_files'] = [
            dataclasses.asdict(action) for action in self.other_files
        ]
        return data

    @classmethod
    def from_dict(cls, d: Mapping) -> 'PositionJob':
        """Job from ``to_dict()`` output."""
        kwargs = dict(d)
        kwargs['channels'] = tuple(
            ChannelSource(**channel) for channel in kwargs['channels']
        )
        kwargs['metadata_csv'] = MetadataCsvSpec.from_dict(
            kwargs['metadata_csv']
        )
        kwargs['time_range'] = tuple(kwargs['time_range'])
        kwargs['physical_size_zyx_um'] = tuple(
            kwargs['physical_size_zyx_um']
        )
        kwargs['other_files'] = tuple(
            OtherFileAction(**action)
            for action in kwargs.get('other_files', ())
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class SkippedPosition:
    """An input position not selected (progress still advances by
    ``size_c``)."""
    input_id: int
    size_c: int


@dataclass(frozen=True)
class PlanContext:
    """Request-wide planning inputs (built by ``convert.prepare``)."""
    layout: sources.RawLayout
    src_dir: str
    dst_dir: str
    files: tuple[str, ...]                  # ORIGINAL names, natsorted
    on_existing: OnExisting
    existing: tuple[int, ...]
    start_pos_n: int
    use_symlinks: bool
    lazy_load: bool
    other_files: Literal['leave', 'move', 'copy']
    one_per_channel: sources.OnePerChannelIndex | None
    mutations: SourceMutations
    source_listing: tuple[str, ...]         # ORIGINAL src listing

    def __post_init__(self) -> None:
        _require_absolute(self.src_dir, 'PlanContext.src_dir')
        _require_absolute(self.dst_dir, 'PlanContext.dst_dir')


def plan_unit_positions(
        ctx: PlanContext, unit: sources.MetadataUnit,
        raw: metadata.RawFileMetadata,
        settings: metadata.ConversionSettings, *, xml_ref: str
    ) -> tuple[list[PositionJob], list[SkippedPosition]]:
    """Plan the Positions of one metadata unit.

    Input ids: single-multi-pos 1..``settings.size_s`` (series = id - 1);
    one-per-pos the file ordinal; one-per-channel the numbers in the file
    names. Output ``Position_<n>``: ``start_pos_n + id - 1`` (layouts 0 and
    1), ``id + offset`` with ``offset = start_pos_n - min(ids)`` for
    CREATE_NEW, else 0 (layout 2). Padding from ``num_positions``.

    Frames: ``settings.frames(raw.size_t)``, checked against the file's
    SizeT when known. Data SizeZ: the file's (layouts 0 and 1, the dialog
    value when unknown), read per raw file by the writer (layout 2).

    Raises
    ------
    PlanError
        Selected input ids that do not exist.
    SettingsError
        Time range outside the file; one file per channel with SizeC above
        the number of channel tokens.
    MissingChannelFilesError
        One file per channel: files of saved (position, channel) pairs are
        missing (all of them listed).
    FilenamePatternError
        One file per channel with renames: the renamed names do not parse
        and the renames change the basename (``_renamed_basename``).
    """
    layout = ctx.layout
    opc = ctx.one_per_channel
    is_one_per_channel = layout == sources.RawLayout.ONE_FILE_PER_CHANNEL
    if is_one_per_channel and opc is None:
        raise ValueError('PlanContext.one_per_channel is required for the '
                         'one-file-per-channel layout')

    if layout == sources.RawLayout.SINGLE_FILE_MULTI_POS:
        input_ids = list(range(1, settings.size_s + 1))
        available = set(input_ids)
    elif layout == sources.RawLayout.ONE_FILE_PER_POS:
        input_ids = [unit.file_ordinal]
        available = set(range(1, len(ctx.files) + 1))
    else:
        input_ids = list(opc.position_ids)
        available = set(input_ids)
    selected = settings.selected_positions
    if selected is not None:
        unknown = sorted(selected - available)
        if unknown:
            raise errors.PlanError(
                'Unknown positions: '
                f'{", ".join(str(pos_id) for pos_id in unknown)} '
                f'(available: {_ranges_text(sorted(available))}).',
                hint='Select existing input positions with --positions.'
            )

    t0, t1 = settings.frames(raw.size_t)
    metadata.validate_time_range(
        t0, t1, raw.size_t, flag=f'time range of {unit.label}'
    )
    if is_one_per_channel and settings.size_c > len(opc.channel_tokens):
        raise errors.SettingsError(
            'One file per channel: SizeC cannot exceed the '
            f'{len(opc.channel_tokens)} channels found in the file names.'
        )

    n_pos = num_positions(
        layout, settings, num_files=len(ctx.files), one_per_channel=opc
    )
    if is_one_per_channel:
        stem_source = opc.basename
        if ctx.mutations.renames:
            stem_source = _renamed_basename(ctx)
        offset = 0
        if ctx.on_existing == OnExisting.CREATE_NEW:
            offset = ctx.start_pos_n - min(opc.position_ids)
    else:
        unit_raw_path = ctx.mutations.final_paths[unit.metadata_file]
        stem_source = os.path.basename(unit_raw_path)
        offset = ctx.start_pos_n - 1
    filename_no_ext = os.path.splitext(stem_source)[0]
    stem = naming.output_stem(stem_source)
    if ctx.use_symlinks:
        output = 'symlink'
    else:
        output = 'h5' if settings.to_h5 else 'tif'

    jobs = []
    skipped = []
    missing = []
    for input_id in input_ids:
        if selected is not None and input_id not in selected:
            skipped.append(SkippedPosition(input_id, settings.size_c))
            continue
        if layout == sources.RawLayout.SINGLE_FILE_MULTI_POS:
            series = input_id - 1
        else:
            series = 0
        output_number = input_id + offset
        pos = naming.pos_str(output_number, n_pos)
        images_path = os.path.join(
            ctx.dst_dir, f'Position_{output_number}', IMAGES_FOLDERNAME
        )
        image_name = ''
        if settings.add_image_name and series < len(raw.scene_names):
            image_name = naming.sanitize_name(raw.scene_names[series] or '')
        basename = naming.position_basename(stem, pos, image_name)

        if is_one_per_channel:
            channels, channel_files, position_missing = (
                _one_per_channel_sources(ctx, settings, input_id)
            )
            missing.extend(position_missing)
            source_file = (
                channel_files[0] if channel_files else unit.metadata_file
            )
            other_files = _other_file_actions(
                ctx, input_id, images_path, filename_no_ext, pos
            )
        else:
            channels = [
                ChannelSource(settings.channel_names[c], c, unit_raw_path, c)
                for c in range(settings.size_c) if settings.save_channels[c]
            ]
            source_file = unit.metadata_file
            other_files = []

        if is_one_per_channel:
            size_z = None
        else:
            size_z = raw.size_z if raw.size_z is not None else settings.size_z
        jobs.append(PositionJob(
            unit_index=unit.index,
            input_id=input_id,
            output_number=output_number,
            num_positions=n_pos,
            pos_str=pos,
            images_path=images_path,
            clear_images=ctx.on_existing == OnExisting.OVERWRITE,
            basename=basename,
            filename_no_ext=filename_no_ext,
            series=series,
            source_file=source_file,
            channels=tuple(channels),
            metadata_csv=metadata_csv_spec(
                settings, basename=basename, saved_size_t=t1 - t0 + 1
            ),
            metadata_xml_ref=xml_ref,
            time_range=(t0, t1),
            size_z=size_z,
            output=output,
            lazy_load=ctx.lazy_load,
            time_increment_s=settings.time_increment_s,
            physical_size_zyx_um=(
                settings.physical_size_z_um, settings.physical_size_y_um,
                settings.physical_size_x_um
            ),
            image_name=image_name,
            other_files=tuple(other_files),
        ))

    if missing:
        missing_text = '\n'.join(
            f'  - position {pos_id}, channel {token}'
            for pos_id, token in missing
        )
        raise errors.MissingChannelFilesError(
            'One file per channel: the files of these saved channels are '
            f'missing:\n{missing_text}',
            missing=missing,
            hint='Add the missing files or do not save those channels.'
        )
    return jobs, skipped


def _renamed_basename(ctx: PlanContext) -> str:
    """One file per channel with renames (``--rename-invalid``): the
    basename of the renamed names, so that the outputs are named as when
    the files are renamed first (GUI rename button).

    A rename can break the pattern: 'exp_2_ch1+2.czi' -> 'exp_2_ch1_2.czi'
    reads as position 1, channel '2'. Positions and channels always come
    from the original names; when the renamed names do not parse, the
    original basename is kept if the renames leave it unchanged (every
    renamed name still contains it).

    Raises
    ------
    FilenamePatternError
        The renamed names do not parse and the renames change the
        basename.
    """
    opc = ctx.one_per_channel
    renamed = [
        os.path.basename(ctx.mutations.final_paths[filename])
        for filename in ctx.files
    ]
    try:
        return sources.parse_one_per_channel_filenames(renamed).basename
    except errors.FilenamePatternError as err:
        if all(opc.basename in name for name in renamed):
            return opc.basename
        pairs = '\n'.join(
            f'  - {old} -> {new}'
            for old, new in zip(ctx.files, renamed) if old != new
        )
        raise errors.FilenamePatternError(
            'The names that --rename-invalid would give the files cannot be '
            'read as one file per channel '
            '(<basename><position number>_<channel>.<ext>), so the outputs '
            f'cannot be named after them:\n{pairs}\n'
            f'With the new names: {err.message}',
            files=ctx.files, hint='Rename the files yourself.',
            details=err.details
        ) from None


def _one_per_channel_sources(
        ctx: PlanContext, settings: metadata.ConversionSettings,
        input_id: int
    ) -> tuple[list[ChannelSource], list[str], list[tuple[int, str]]]:
    """Saved channels of one position (dialog row c reads the file of raw
    token c), their ORIGINAL file names and the missing (position, token)
    pairs."""
    tokens = ctx.one_per_channel.channel_tokens
    channels = []
    channel_files = []
    missing = []
    for c in range(settings.size_c):
        if not settings.save_channels[c]:
            continue
        key = (input_id, tokens[c])
        filename = ctx.one_per_channel.files.get(key)
        if filename is None:
            missing.append(key)
            continue
        channels.append(ChannelSource(
            settings.channel_names[c], c, ctx.mutations.final_paths[filename],
            0
        ))
        channel_files.append(filename)
    return channels, channel_files, missing


def _other_file_actions(
        ctx: PlanContext, input_id: int, images_path: str,
        filename_no_ext: str, pos: str
    ) -> list[OtherFileAction]:
    """Files of the ORIGINAL source listing named
    ``<basename><input_id>_<rest>`` (zero padding allowed) that are not raw
    channel files (any file whose name parses to a (position, channel) of
    the index, whatever its extension), moved/copied to
    ``<filename_no_ext>_s<pos>_<rest>``."""
    if ctx.other_files == 'leave':
        return []
    opc = ctx.one_per_channel
    pattern = re.compile(rf'^{re.escape(opc.basename)}0*{input_id}_(.+)$')
    actions = []
    for name in ctx.source_listing:
        parsed = sources.split_position_channel(naming.raw_file_stem(name))
        if parsed is not None and (int(parsed[0]), parsed[1]) in opc.files:
            continue
        match = pattern.match(name)
        if match is None:
            continue
        actions.append(OtherFileAction(
            src=os.path.join(ctx.src_dir, name),
            dst=os.path.join(
                images_path, f'{filename_no_ext}_s{pos}_{match.group(1)}'
            ),
            mode=ctx.other_files,
        ))
    return actions


def _ranges_text(ids: Iterable[int]) -> str:
    """'1-3, 5' for [1, 2, 3, 5]."""
    spans: list[list[int]] = []
    for pos_id in sorted(ids):
        if spans and spans[-1][1] == pos_id - 1:
            spans[-1][1] = pos_id
        else:
            spans.append([pos_id, pos_id])
    return ', '.join(
        str(first) if first == last else f'{first}-{last}'
        for first, last in spans
    ) or 'none'


# ------------------------------------------------------------- conflicts
def find_conflicts(
        jobs: Iterable[PositionJob], existing: Sequence[int],
        on_existing: OnExisting
    ) -> list[int]:
    """Sorted planned output numbers that already exist and must not be
    written into: every one for ERROR and CREATE_NEW, none for OVERWRITE
    and ADD (writing into them is what the user asked for)."""
    if on_existing in (OnExisting.OVERWRITE, OnExisting.ADD):
        return []
    planned = {job.output_number for job in jobs}
    return sorted(planned.intersection(existing))


def check_conflicts(
        jobs: Iterable[PositionJob], existing: Sequence[int],
        on_existing: OnExisting
    ) -> None:
    """Raise ``ExistingPositionsError`` when ``find_conflicts`` is not
    empty, listing the Position numbers and the choices."""
    conflicts = find_conflicts(jobs, existing, on_existing)
    if not conflicts:
        return
    folders = ', '.join(f'Position_{n}' for n in conflicts)
    raise errors.ExistingPositionsError(
        f'The destination already contains these Position folders: '
        f'{folders}.',
        existing=conflicts, choices=ON_EXISTING_CHOICES,
        hint=(
            'Choose what to do with --on-existing: overwrite (DELETES '
            'Position_N/Images of every position written), add (write into '
            'the existing folders) or create-new (new Position folders '
            'after the last one).'
        )
    )
