"""Headless CLI for ACDC data structure conversion and metadata extraction."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from . import io, load, myutils

LAYOUT_TO_RAW_DATA_STRUCT = {
    'single-multi-pos': 0,
    'one-per-pos': 1,
    'one-per-channel': 2,
}

RAW_DATA_STRUCT_TO_LAYOUT = {v: k for k, v in LAYOUT_TO_RAW_DATA_STRUCT.items()}

RESTRUCTURE_LAYOUTS = (
    'multi-timepoint',
    'multi-channel',
)

# Frame number must be at the end with .ext, e.g., _t01.tif
FRAME_NAME_PATTERNS = (
    r'_(day)?(\d+)\.[A-Za-z0-9]+$',
    r'_(t)?(\d+)\.[A-Za-z0-9]+$',
)


def get_frame_num_and_pattern(filename):
    matching_frame_name_pattern = r'^\.+'
    frame_number = None
    for frame_name_pattern in FRAME_NAME_PATTERNS:
        try:
            frame_number = re.findall(frame_name_pattern, filename)[0][1]
            matching_frame_name_pattern = frame_name_pattern
            break
        except Exception:
            frame_number = None
    return matching_frame_name_pattern, frame_number


def read_filename_pattern(file_name):
    matching_frame_name_pattern, frame_number = get_frame_num_and_pattern(
        file_name
    )
    s = re.sub(matching_frame_name_pattern, '', file_name)
    for i, c in enumerate(s[::-1]):
        if c == '_':
            break
    channel_name = s[-i:]
    pos_name = s[:-i - 1]
    if channel_name.endswith('.tif'):
        channel_name = channel_name[:-4]
    return pos_name, frame_number, channel_name


@dataclass
class HeadlessMetadataHandler:
    trust_metadata: bool = True
    to_h5: bool = False
    lens_na: Optional[float] = None
    size_t: Optional[int] = None
    size_z: Optional[int] = None
    size_c: Optional[int] = None
    size_s: Optional[int] = None
    time_increment: Optional[float] = None
    physical_size_x: Optional[float] = None
    physical_size_y: Optional[float] = None
    physical_size_z: Optional[float] = None
    channels: Optional[list] = None
    em_wavelens: Optional[list] = None
    positions: Optional[list] = None
    time_range_start: int = 0
    time_range_end: Optional[int] = None
    save_channels: Optional[list] = None
    add_image_name: bool = False
    basename: Optional[str] = None

    @classmethod
    def from_parser_args(cls, args: dict) -> 'HeadlessMetadataHandler':
        channels = None
        if args.get('channels'):
            channels = [c.strip() for c in args['channels'].split(',') if c.strip()]

        em_wavelens = None
        if args.get('em_wavelens'):
            em_wavelens = [
                float(w.strip()) for w in args['em_wavelens'].split(',') if w.strip()
            ]

        positions = None
        if args.get('positions'):
            positions = [p.strip() for p in args['positions'].split(',') if p.strip()]

        save_channels = None
        if args.get('save_channels'):
            save_channels = [
                s.strip().lower() in ('1', 'true', 'yes')
                for s in args['save_channels'].split(',')
            ]

        time_range_start, time_range_end = parse_time_range(args.get('time_range'))

        handler = cls(
            trust_metadata=args.get('trust_metadata', True),
            to_h5=args.get('format', 'tif') == 'h5',
            lens_na=args.get('lens_na'),
            size_t=args.get('size_t'),
            size_z=args.get('size_z'),
            size_c=args.get('size_c'),
            size_s=args.get('size_s'),
            time_increment=args.get('time_increment'),
            physical_size_x=args.get('physical_size_x'),
            physical_size_y=args.get('physical_size_y'),
            physical_size_z=args.get('physical_size_z'),
            channels=channels,
            em_wavelens=em_wavelens,
            positions=positions,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            save_channels=save_channels,
            add_image_name=args.get('add_image_name', False),
            basename=args.get('basename'),
        )

        metadata_csv = args.get('metadata_csv')
        if metadata_csv:
            handler.apply_metadata_csv(metadata_csv)

        return handler

    def apply_metadata_csv(self, metadata_csv_path: str):
        df = pd.read_csv(metadata_csv_path).set_index('Description')
        field_map = {
            'LensNA': ('lens_na', float),
            'SizeT': ('size_t', int),
            'SizeZ': ('size_z', int),
            'TimeIncrement': ('time_increment', float),
            'PhysicalSizeX': ('physical_size_x', float),
            'PhysicalSizeY': ('physical_size_y', float),
            'PhysicalSizeZ': ('physical_size_z', float),
            'basename': ('basename', str),
        }
        for csv_key, (attr, cast) in field_map.items():
            if csv_key in df.index:
                setattr(self, attr, cast(df.at[csv_key, 'values']))

        ch_names = []
        em_wavelens = []
        c = 0
        while f'channel_{c}_name' in df.index:
            ch_names.append(str(df.at[f'channel_{c}_name', 'values']))
            wavelen_key = f'channel_{c}_emWavelen'
            if wavelen_key in df.index:
                em_wavelens.append(float(df.at[wavelen_key, 'values']))
            c += 1
        if ch_names:
            self.channels = ch_names
        if em_wavelens:
            self.em_wavelens = em_wavelens


@dataclass
class ParsedRawMetadata:
    lens_na: float = 1.4
    size_t: int = 1
    size_z: int = 1
    size_c: int = 1
    size_s: int = 1
    time_increment: float = 1.0
    time_increment_unit: str = 's'
    physical_size_x: float = 1.0
    physical_size_y: float = 1.0
    physical_size_z: float = 1.0
    physical_size_unit: str = 'μm'
    channel_names: list = field(default_factory=list)
    em_wavelens: list = field(default_factory=list)
    image_name: str = ''
    metadata_xml: str = ''


def parse_time_range(time_range: Optional[str]) -> tuple:
    if not time_range:
        return 0, None
    if ':' in time_range:
        start, end = time_range.split(':', 1)
        return int(start), int(end)
    return 0, int(time_range)


def layout_to_raw_data_struct(layout: str) -> int:
    try:
        return LAYOUT_TO_RAW_DATA_STRUCT[layout]
    except KeyError as err:
        valid = ', '.join(LAYOUT_TO_RAW_DATA_STRUCT)
        raise ValueError(
            f'Invalid layout "{layout}". Valid values: {valid}'
        ) from err


def on_existing_to_worker_flags(on_existing: str) -> dict:
    if on_existing == 'overwrite':
        return {'overwrite': True, 'add_files': False, 'create_new': False}
    if on_existing == 'add':
        return {'overwrite': False, 'add_files': True, 'create_new': False}
    if on_existing == 'create-new':
        return {'overwrite': False, 'add_files': False, 'create_new': True}
    raise ValueError(
        f'Invalid on-existing policy "{on_existing}". '
        'Valid values: overwrite, add, create-new'
    )


def list_raw_microscopy_files(raw_src_path: str, layout: str) -> list:
    ls = natsorted(myutils.listdir(raw_src_path))
    files = [
        filename for filename in ls
        if os.path.isfile(os.path.join(raw_src_path, filename))
    ]
    if not files:
        raise FileNotFoundError(
            f'No files found in input folder "{raw_src_path}"'
        )

    extensions = [
        os.path.splitext(filename)[1] for filename in files
    ]
    unique_ext = list(dict.fromkeys(extensions))
    if len(unique_ext) > 1:
        from collections import Counter
        most_common_ext, _ = Counter(extensions).most_common(1)[0]
        files = [
            filename for filename in files
            if os.path.splitext(filename)[1] == most_common_ext
        ]

    if layout == 'single-multi-pos' and len(files) > 1:
        raise ValueError(
            'Layout "single-multi-pos" expects a single microscopy file in the '
            f'input folder, but found {len(files)} files: {files}'
        )

    return files


def parse_one_per_channel_files(raw_filenames: list) -> tuple:
    ch_names = set()
    pos_nums = set()
    stripped_filenames = []
    for file in raw_filenames:
        filename, _ = os.path.splitext(file)
        m_iter = myutils.findalliter(r'(\d+)_(.+)', filename)
        if len(m_iter) <= 1:
            raise ValueError(
                'Files for layout "one-per-channel" must match the pattern '
                'basenameN_channelName (e.g. ASY015_1_GFP). '
                f'Could not parse filename "{file}".'
            )
        m = m_iter[-2]
        pos_num, ch_name = int(m[0][0]), m[0][1]
        ch_names.add(ch_name)
        pos_nums.add(pos_num)
        ch_idx = filename.find(f'{pos_num}_{ch_name}')
        stripped_filenames.append(filename[:ch_idx])

    basename = myutils.getBasename(stripped_filenames)
    if not basename:
        raise ValueError(
            'Could not determine common basename from one-per-channel filenames.'
        )

    return basename, sorted(pos_nums), sorted(ch_names, key=str)


def read_metadata_bioio(raw_filepath: str):
    from . import bioio_sample_data_folderpath, _process
    from . import acdc_bioio_bioformats as bioformats
    import subprocess

    read_metadata_py_filepath = os.path.join(
        os.path.dirname(bioformats.__file__), '_read_metadata.py'
    )
    uuid4 = uuid.uuid4()
    command = (
        f'{sys.executable}, {read_metadata_py_filepath}, '
        f'-f, {raw_filepath}, '
        f'-uuid, {uuid4}'
    )
    args = [sys.executable, _process.__file__, '-c', command]
    subprocess.run(args, check=False)
    bioformats._utils.check_raise_exception(uuid4)

    metadataXML_filepath = os.path.join(
        bioio_sample_data_folderpath, 'metadataXML.txt'
    )
    metadataXML = bioformats.Metadata().init_from_file(metadataXML_filepath)

    metadata_filepath = os.path.join(
        bioio_sample_data_folderpath, 'metadata.txt'
    )
    metadata = bioformats.OMEXML().init_from_file(
        metadata_filepath, raw_filepath
    )
    return metadata, metadataXML


def parse_raw_metadata(raw_filepath: str) -> ParsedRawMetadata:
    from . import load

    if raw_filepath.endswith('.ome.tif'):
        metadata = load.OMEXML(raw_filepath)
        metadata_xml = metadata.omexml_string
    else:
        metadata, metadata_xml_obj = read_metadata_bioio(raw_filepath)
        metadata_xml = str(metadata_xml_obj)

    parsed = ParsedRawMetadata(metadata_xml=metadata_xml)

    try:
        parsed.lens_na = float(metadata.instrument().Objective.LensNA)
    except Exception:
        pass

    try:
        parsed.size_s = int(metadata.get_image_count())
    except Exception:
        pass

    try:
        parsed.size_z = int(metadata.image().Pixels.SizeZ)
    except Exception:
        pass

    try:
        parsed.size_t = int(metadata.image().Pixels.SizeT)
    except Exception:
        pass

    try:
        parsed.time_increment = float(metadata.image().Pixels.node.get('TimeIncrement'))
    except Exception:
        pass

    try:
        unit = metadata.image().Pixels.node.get('TimeIncrementUnit')
        if unit is not None:
            parsed.time_increment_unit = unit
    except Exception:
        pass

    try:
        parsed.size_c = int(metadata.image().Pixels.SizeC)
    except Exception:
        pass

    try:
        parsed.physical_size_x = float(metadata.image().Pixels.PhysicalSizeX)
    except Exception:
        pass

    try:
        parsed.physical_size_y = float(metadata.image().Pixels.PhysicalSizeY)
    except Exception:
        pass

    try:
        parsed.physical_size_z = float(metadata.image().Pixels.PhysicalSizeZ)
    except Exception:
        pass

    try:
        unit = metadata.image().Pixels.node.get('PhysicalSizeXUnit')
        if unit is not None:
            parsed.physical_size_unit = unit
    except Exception:
        pass

    try:
        image_name = metadata.image().Name
        if image_name is not None:
            parsed.image_name = image_name
    except Exception:
        pass

    ch_names = []
    em_wavelens = []
    for c in range(parsed.size_c):
        try:
            ch_names.append(metadata.image().Pixels.Channel(c).Name or f'channel_{c}')
        except Exception:
            ch_names.append(f'channel_{c}')
        try:
            em_wavelen = metadata.image().Pixels.Channel(c).node.get('EmissionWavelength')
            em_wavelens.append(float(em_wavelen))
        except Exception:
            em_wavelens.append(500.0)

    parsed.channel_names = ch_names
    parsed.em_wavelens = em_wavelens
    return parsed


def metadata_to_dataframe(
        parsed: ParsedRawMetadata,
        basename: str,
    ) -> pd.DataFrame:
    df = pd.DataFrame({
        'LensNA': parsed.lens_na,
        'SizeT': parsed.size_t,
        'SizeZ': parsed.size_z,
        'TimeIncrement': parsed.time_increment,
        'PhysicalSizeZ': parsed.physical_size_z,
        'PhysicalSizeY': parsed.physical_size_y,
        'PhysicalSizeX': parsed.physical_size_x,
        'basename': basename,
    }, index=['values']).T
    df.index.name = 'Description'

    ch_metadata = list(parsed.channel_names)
    ch_metadata.extend(parsed.em_wavelens)
    description = [f'channel_{c}_name' for c in range(len(parsed.channel_names))]
    description.extend([
        f'channel_{c}_emWavelen' for c in range(len(parsed.channel_names))
    ])
    df_channel_names = pd.DataFrame({
        'Description': description,
        'values': ch_metadata,
    }).set_index('Description')
    return pd.concat([df, df_channel_names])


def guess_basename_from_filepath(raw_filepath: str) -> str:
    filename = os.path.splitext(os.path.basename(raw_filepath))[0]
    return f'{filename}_'


def init_bioio_reader(raw_filepath: str, logger_func=print):
    from . import acdc_bioio_bioformats as bioformats
    import subprocess
    from . import _process

    bioformats.install.install_reader_dependencies(
        raw_filepath,
        exception=Exception(
            'Failed installing reader dependencies from the CLI.'
        ),
    )

    init_reader_py_filepath = os.path.join(
        os.path.dirname(bioformats.__file__), '_init_reader.py'
    )
    uuid4 = uuid.uuid4()
    command = (
        f'{sys.executable}, {init_reader_py_filepath}, '
        f'-f, {raw_filepath}, '
        f'-uuid, {uuid4}'
    )
    args = [sys.executable, _process.__file__, '-c', command]
    subprocess.run(args, check=False)
    bioformats._utils.check_raise_exception(uuid4)
    logger_func('BioIO reader initialized.')


def get_start_pos_n(exp_dst_path: str, on_existing: str) -> int:
    if on_existing != 'create-new':
        return 1
    pos_foldernames = myutils.get_pos_foldernames(exp_dst_path)
    if not pos_foldernames:
        return 1
    pos_ns = [int(pos.split('_')[-1]) for pos in pos_foldernames]
    return max(pos_ns) + 1


def run_metadata_cli(args: dict):
    input_path = os.path.abspath(args['input'])
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input path does not exist: "{input_path}"')

    if os.path.isdir(input_path):
        files = list_raw_microscopy_files(input_path, 'one-per-pos')
        raw_filepath = os.path.join(input_path, files[0])
    else:
        raw_filepath = input_path

    parsed = parse_raw_metadata(raw_filepath)
    basename = args.get('basename') or guess_basename_from_filepath(raw_filepath)
    df = metadata_to_dataframe(parsed, basename)

    output_format = args.get('format', 'text')
    if output_format == 'json':
        payload = asdict(parsed)
        payload['basename'] = basename
        print(json.dumps(payload, indent=2))
        return

    if output_format == 'csv':
        print(df.to_csv())
        return

    print(f'File: {raw_filepath}')
    print(f'basename: {basename}')
    for idx, row in df.iterrows():
        print(f'{idx}: {row["values"]}')

    output_dir = args.get('output')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metadata_csv_path = os.path.join(output_dir, f'{basename}metadata.csv')
        df.to_csv(metadata_csv_path)
        metadata_xml_path = os.path.join(output_dir, f'{basename}metadataXML.txt')
        with open(metadata_xml_path, 'w', encoding='utf-8') as txt:
            txt.write(parsed.metadata_xml)
        print(f'Wrote metadata to "{metadata_csv_path}"')


@dataclass
class ConvertState:
    lens_na: float
    size_t: int
    size_z: int
    size_c: int
    size_s: int
    time_increment: float
    physical_size_x: float
    physical_size_y: float
    physical_size_z: float
    channel_names: list
    em_wavelens: list
    metadata_xml: str
    to_h5: bool = False
    selected_pos: list = field(default_factory=lambda: ['All Positions'])
    time_range_start: int = 0
    time_range_end: Optional[int] = None
    save_channels: Optional[list] = None
    add_image_name: bool = False
    image_name: str = ''


def build_convert_state(
        parsed: ParsedRawMetadata,
        handler: HeadlessMetadataHandler,
    ) -> ConvertState:
    lens_na = handler.lens_na if handler.lens_na is not None else parsed.lens_na
    size_t = handler.size_t if handler.size_t is not None else parsed.size_t
    size_z = handler.size_z if handler.size_z is not None else parsed.size_z
    size_c = handler.size_c if handler.size_c is not None else parsed.size_c
    size_s = handler.size_s if handler.size_s is not None else parsed.size_s
    time_increment = (
        handler.time_increment if handler.time_increment is not None
        else parsed.time_increment
    )
    physical_size_x = (
        handler.physical_size_x if handler.physical_size_x is not None
        else parsed.physical_size_x
    )
    physical_size_y = (
        handler.physical_size_y if handler.physical_size_y is not None
        else parsed.physical_size_y
    )
    physical_size_z = (
        handler.physical_size_z if handler.physical_size_z is not None
        else parsed.physical_size_z
    )
    channel_names = handler.channels if handler.channels is not None else parsed.channel_names
    size_c = len(channel_names)
    em_wavelens = (
        handler.em_wavelens if handler.em_wavelens is not None
        else parsed.em_wavelens[:size_c]
    )
    return ConvertState(
        lens_na=lens_na,
        size_t=size_t,
        size_z=size_z,
        size_c=size_c,
        size_s=size_s,
        time_increment=time_increment,
        physical_size_x=physical_size_x,
        physical_size_y=physical_size_y,
        physical_size_z=physical_size_z,
        channel_names=channel_names,
        em_wavelens=em_wavelens,
        metadata_xml=parsed.metadata_xml,
        to_h5=handler.to_h5,
        selected_pos=(
            handler.positions if handler.positions is not None
            else ['All Positions']
        ),
        time_range_start=handler.time_range_start,
        time_range_end=(
            handler.time_range_end if handler.time_range_end is not None
            else size_t - 1
        ),
        save_channels=handler.save_channels,
        add_image_name=handler.add_image_name,
        image_name=parsed.image_name,
    )


def _sanitize_image_name(image_name: str) -> str:
    ch_name = "".join(
        c if c.isalnum() or c == '_' or c == '' else '_' for c in image_name
    )
    while ch_name.endswith('_'):
        ch_name = ch_name[:-1]
    return ch_name


def get_acdc_filename(
        filename_no_ext: str,
        s0p: str,
        append_txt: str,
        ext: str,
        add_image_name: bool = False,
        image_name: str = '',
        return_basename: bool = False,
    ):
    filename_no_ext = filename_no_ext.replace('.', '_')
    if add_image_name and image_name:
        image_name = _sanitize_image_name(image_name)
        basename = f'{filename_no_ext}_{image_name}_s{s0p}_'
    else:
        basename = f'{filename_no_ext}_s{s0p}_'
    filename = f'{basename}{append_txt}{ext}'
    if return_basename:
        return filename, basename
    return filename


def _write_position_metadata(
        state: ConvertState,
        images_path: str,
        filename_no_ext: str,
        s0p: str,
        series: int,
    ) -> str:
    metadata_xml_path = os.path.join(
        images_path,
        get_acdc_filename(
            filename_no_ext, s0p, 'metadataXML', '.txt',
            state.add_image_name, state.image_name,
        ),
    )
    with open(metadata_xml_path, 'w', encoding='utf-8') as txt:
        txt.write(state.metadata_xml)

    metadata_filename, basename = get_acdc_filename(
        filename_no_ext, s0p, 'metadata', '.csv',
        state.add_image_name, state.image_name,
        return_basename=True,
    )
    metadata_csv_path = os.path.join(images_path, metadata_filename)
    saved_size_t = state.time_range_end - state.time_range_start + 1
    save_channels = state.save_channels or [True] * state.size_c
    df = pd.DataFrame({
        'LensNA': state.lens_na,
        'SizeT': saved_size_t,
        'SizeZ': state.size_z,
        'TimeIncrement': state.time_increment,
        'PhysicalSizeZ': state.physical_size_z,
        'PhysicalSizeY': state.physical_size_y,
        'PhysicalSizeX': state.physical_size_x,
        'basename': basename,
    }, index=['values']).T
    df.index.name = 'Description'

    ch_metadata = [
        ch_name for c, ch_name in enumerate(state.channel_names)
        if save_channels[c]
    ]
    description = [
        f'channel_{c}_name' for c in range(state.size_c) if save_channels[c]
    ]
    ch_metadata.extend([
        wavelen for c, wavelen in enumerate(state.em_wavelens)
        if save_channels[c]
    ])
    description.extend([
        f'channel_{c}_emWavelen' for c in range(state.size_c)
        if save_channels[c]
    ])
    df = pd.concat([
        df,
        pd.DataFrame({
            'Description': description,
            'values': ch_metadata,
        }).set_index('Description'),
    ])
    df.to_csv(metadata_csv_path)
    return basename


def _run_bioio_subprocess(command: str, uuid4):
    from . import _process
    from . import acdc_bioio_bioformats as bioformats

    args = [sys.executable, _process.__file__, '-c', command]
    subprocess.run(args, check=False)
    bioformats._utils.check_raise_exception(uuid4)


def _save_channels_bioio(
        state: ConvertState,
        raw_filepath: str,
        images_path: str,
        filename_no_ext: str,
        s0p: str,
        series: int,
        lazy_load: bool,
        logger_func=print,
    ):
    from . import acdc_bioio_bioformats as bioformats

    save_data_py_filepath = os.path.join(
        os.path.dirname(bioformats.__file__), '_save_data.py'
    )
    save_channels = state.save_channels or [True] * state.size_c
    zyx_physical_sizes = " ".join([
        str(state.physical_size_z),
        str(state.physical_size_y),
        str(state.physical_size_x),
    ])
    uuid4 = uuid.uuid4()
    command = (
        f'{sys.executable}, {save_data_py_filepath}, '
        f'-f, {raw_filepath}, '
        f'-d, {" ".join([str(val) for val in save_channels])}, '
        f'-c, {" ".join(state.channel_names)}, '
        f'-s, {series}, '
        f'-i, {images_path}, '
        f'-p, {filename_no_ext}, '
        f'-pos, {s0p}, '
        f'-t, {state.size_t}, '
        f'-z, {state.size_z}, '
        f'-time_increment, {state.time_increment}, '
        f'-zyx, {zyx_physical_sizes}, '
        f'-r, {state.time_range_start} {state.time_range_end}, '
        f'-uuid, {uuid4}'
    )
    if state.to_h5:
        command = f'{command}, -to_h5'
    if not lazy_load:
        command = f'{command}, -a'
    logger_func(
        f'Saving channels via BioIO for series {series} to {images_path}...'
    )
    _run_bioio_subprocess(command, uuid4)


def _save_single_channel_bioio(
        state: ConvertState,
        raw_filepath: str,
        images_path: str,
        filename_no_ext: str,
        s0p: str,
        series: int,
        ch_name: str,
        ch_idx: int,
        lazy_load: bool,
        logger_func=print,
    ):
    from . import acdc_bioio_bioformats as bioformats

    save_data_py_filepath = os.path.join(
        os.path.dirname(bioformats.__file__), '_save_data_single_channel.py'
    )
    save_channels = state.save_channels or [True] * state.size_c
    zyx_physical_sizes = " ".join([
        str(state.physical_size_z),
        str(state.physical_size_y),
        str(state.physical_size_x),
    ])
    uuid4 = uuid.uuid4()
    command = (
        f'{sys.executable}, {save_data_py_filepath}, '
        f'-f, {raw_filepath}, '
        f'-d, {" ".join([str(val) for val in save_channels])}, '
        f'-c, {ch_name}, '
        f'-ch_idx, {ch_idx}, '
        f'-s, {series}, '
        f'-i, {images_path}, '
        f'-p, {filename_no_ext}, '
        f'-pos, {s0p}, '
        f'-t, {state.size_t}, '
        f'-z, {state.size_z}, '
        f'-time_increment, {state.time_increment}, '
        f'-zyx, {zyx_physical_sizes}, '
        f'-r, {state.time_range_start} {state.time_range_end}, '
        f'-uuid, {uuid4}'
    )
    if state.to_h5:
        command = f'{command}, -to_h5'
    if not lazy_load:
        command = f'{command}, -a'
    logger_func(f'Saving channel {ch_name} via BioIO...')
    _run_bioio_subprocess(command, uuid4)


def _should_save_position(state: ConvertState, in_file_pos_idx: int) -> bool:
    in_file_pos_name = f'Position_{in_file_pos_idx + 1}'
    return (
        'All Positions' in state.selected_pos
        or in_file_pos_name in state.selected_pos
    )


def _save_to_pos_folder(
        state: ConvertState,
        raw_src_path: str,
        exp_dst_path: str,
        filename: str,
        series: int,
        pos_n: int,
        num_pos_digits: int,
        raw_data_struct: int,
        overwrite_pos: bool,
        create_new: bool,
        lazy_load: bool,
        logger_func=print,
        basename_for_channels: Optional[str] = None,
    ):
    raw_filepath = os.path.join(raw_src_path, filename)
    if not _should_save_position(state, series):
        return

    pos_path = os.path.join(exp_dst_path, f'Position_{pos_n}')
    images_path = os.path.join(pos_path, 'Images')

    if os.path.exists(images_path) and overwrite_pos:
        shutil.rmtree(images_path)

    if os.path.exists(images_path) and create_new:
        images_path = re.sub(
            r'Position_\d+', f'Position_{pos_n}', images_path
        )

    os.makedirs(images_path, exist_ok=True)
    s0p = str(pos_n).zfill(num_pos_digits)
    filename_no_ext, _ = os.path.splitext(filename)

    logger_func(
        f'Position {pos_n}: saving data to {images_path}...'
    )
    _write_position_metadata(
        state, images_path, filename_no_ext, s0p, series
    )

    if raw_data_struct != 2:
        _save_channels_bioio(
            state, raw_filepath, images_path, filename_no_ext, s0p,
            series, lazy_load, logger_func=logger_func,
        )
    else:
        save_channels = state.save_channels or [True] * state.size_c
        channel_basename = basename_for_channels or filename_no_ext
        for c, (ch_name, save_ch) in enumerate(
            zip(state.channel_names, save_channels)
        ):
            if not save_ch:
                continue
            raw_filename = f'{channel_basename}{pos_n}_{ch_name}'
            channel_raw_filepath = next(
                os.path.join(raw_src_path, f)
                for f in myutils.listdir(raw_src_path)
                if f.find(raw_filename) != -1
            )
            _save_single_channel_bioio(
                state, channel_raw_filepath, images_path, filename_no_ext,
                s0p, series, ch_name, c, lazy_load, logger_func=logger_func,
            )


def _move_raw_file(raw_src_path: str, filename: str, move_raw: bool):
    if not move_raw:
        return
    if os.path.basename(raw_src_path) == 'raw_microscopy_files':
        return
    raw_filepath = os.path.join(raw_src_path, filename)
    raw_path = os.path.join(raw_src_path, 'raw_microscopy_files')
    os.makedirs(raw_path, exist_ok=True)
    dst = os.path.join(raw_path, filename)
    try:
        shutil.move(raw_filepath, dst)
    except PermissionError as err:
        print(err)


def run_convert_cli(args: dict, logger_func=print):
    raw_src_path = os.path.abspath(args['input'])
    exp_dst_path = os.path.abspath(args['output'])
    layout = args['layout']
    raw_data_struct = layout_to_raw_data_struct(layout)

    if not os.path.isdir(raw_src_path):
        raise NotADirectoryError(
            f'Input path must be a folder containing raw microscopy files: '
            f'"{raw_src_path}"'
        )

    os.makedirs(exp_dst_path, exist_ok=True)
    raw_filenames = list_raw_microscopy_files(raw_src_path, layout)
    logger_func(
        f'Found {len(raw_filenames)} raw file(s) in "{raw_src_path}"'
    )

    on_existing = args.get('on_existing', 'overwrite')
    worker_flags = on_existing_to_worker_flags(on_existing)
    start_pos_n = get_start_pos_n(exp_dst_path, on_existing)
    metadata_handler = HeadlessMetadataHandler.from_parser_args(args)
    lazy_load = args.get('lazy_load', True)
    move_raw = args.get('move_raw', True)
    if exp_dst_path == raw_src_path and not move_raw:
        logger_func(
            'Input and output are the same folder; enabling --move-raw.'
        )
        move_raw = True

    raw_filepath = os.path.join(raw_src_path, raw_filenames[0])
    init_bioio_reader(raw_filepath, logger_func=logger_func)

    overwrite_pos = worker_flags['overwrite']
    create_new = worker_flags['create_new']

    channel_basename = metadata_handler.basename
    if raw_data_struct == 2:
        if channel_basename is None:
            channel_basename, pos_nums, channel_names = (
                parse_one_per_channel_files(raw_filenames)
            )
        else:
            _, pos_nums, channel_names = parse_one_per_channel_files(
                raw_filenames
            )
        metadata_handler.channels = metadata_handler.channels or channel_names

    for p, filename in enumerate(raw_filenames):
        pos_n = p + start_pos_n
        parsed = parse_raw_metadata(os.path.join(raw_src_path, filename))
        state = build_convert_state(parsed, metadata_handler)

        if raw_data_struct == 0:
            num_pos = state.size_s
            num_pos_digits = len(str(num_pos))
            for in_file_p in range(state.size_s):
                _save_to_pos_folder(
                    state, raw_src_path, exp_dst_path, filename,
                    in_file_p, pos_n, num_pos_digits, raw_data_struct,
                    overwrite_pos, create_new, lazy_load,
                    logger_func=logger_func,
                )
        elif raw_data_struct == 1:
            num_pos = len(raw_filenames)
            num_pos_digits = len(str(num_pos))
            _save_to_pos_folder(
                state, raw_src_path, exp_dst_path, filename,
                0, pos_n, num_pos_digits, raw_data_struct,
                overwrite_pos, create_new, lazy_load,
                logger_func=logger_func,
            )
        else:
            break

        _move_raw_file(raw_src_path, filename, move_raw)

    if raw_data_struct == 2:
        parsed = parse_raw_metadata(
            os.path.join(raw_src_path, raw_filenames[0])
        )
        state = build_convert_state(parsed, metadata_handler)
        num_pos = len(pos_nums)
        num_pos_digits = len(str(num_pos))
        for p_idx, pos in enumerate(pos_nums):
            _save_to_pos_folder(
                state, raw_src_path, exp_dst_path, channel_basename,
                0, pos, num_pos_digits, raw_data_struct,
                overwrite_pos, create_new, lazy_load,
                logger_func=logger_func,
                basename_for_channels=channel_basename,
            )
        for filename in raw_filenames:
            _move_raw_file(raw_src_path, filename, move_raw)

    logger_func(f'Conversion completed. Output saved to "{exp_dst_path}".')


def list_restructure_files(folder_path: str) -> list:
    ls = natsorted(myutils.listdir(folder_path))
    files = [
        filename for filename in ls
        if os.path.isfile(os.path.join(folder_path, filename))
    ]
    if not files:
        raise FileNotFoundError(
            f'No files found in input folder "{folder_path}"'
        )

    extensions = [os.path.splitext(filename)[1] for filename in files]
    unique_ext = list(dict.fromkeys(extensions))
    if len(unique_ext) > 1:
        most_common_ext, _ = Counter(extensions).most_common(1)[0]
        files = [
            filename for filename in files
            if os.path.splitext(filename)[1] == most_common_ext
        ]
    return files


def restructure_multi_channel(
        src_path: str,
        dst_path: str,
        action: str = 'copy',
        logger_func=print,
    ):
    if action not in ('copy', 'move'):
        raise ValueError('action must be "copy" or "move"')
    load._restructure_multi_files_multi_pos(
        src_path, dst_path, action=action, signals=None, logger=logger_func
    )


def restructure_multi_timepoint(
        src_path: str,
        dst_path: str,
        channels: list,
        basename: str = '',
        segm_folder: str = '',
        logger_func=print,
    ):
    if not channels:
        raise ValueError('At least one channel name is required.')

    valid_filenames = list_restructure_files(src_path)
    sample_filename = valid_filenames[0]
    frame_name_pattern, _ = get_frame_num_and_pattern(sample_filename)

    files_info = {}
    for file in valid_filenames:
        try:
            for ch in channels:
                match = re.findall(rf'(.*)_{re.escape(ch)}{frame_name_pattern}', file)
                if match:
                    break
            else:
                raise FileNotFoundError(
                    f'The file name "{file}" does not contain any channel name'
                )
            pos_name, _, frame_name = match[0]
            frame_number = int(frame_name)
            if pos_name not in files_info:
                files_info[pos_name] = {ch: [(file, frame_number)]}
            elif ch not in files_info[pos_name]:
                files_info[pos_name][ch] = [(file, frame_number)]
            else:
                files_info[pos_name][ch].append((file, frame_number))
        except Exception:
            logger_func(
                f'WARNING: File "{file}" does not contain a valid pattern. '
                'Skipping it.'
            )

    all_pos_data_info = []
    for p, (pos_name, channel_info) in enumerate(files_info.items()):
        logger_func('=' * 40)
        logger_func(f'Processing position "{pos_name}"...')

        img = None
        for files_list in channel_info.values():
            file_path = os.path.join(src_path, files_list[0][0])
            try:
                img = load.imread(file_path)
                break
            except Exception:
                continue
        if img is None:
            logger_func(
                f'WARNING: No valid image files found for position "{pos_name}"'
            )
            continue

        if basename:
            pos_basename = f'{basename}_{pos_name}_'
        else:
            pos_basename = f'{pos_name}_'

        first_files_list = next(iter(channel_info.values()))
        size_t = len(first_files_list)
        df_metadata = pd.DataFrame({
            'SizeT': size_t,
            'basename': pos_basename,
        }, index=['values'])

        for c, (channel_name, files_list) in enumerate(channel_info.items()):
            logger_func(f'  Processing channel "{channel_name}"...')
            sorted_files_list = sorted(files_list, key=lambda t: t[1])
            df_metadata[f'channel_{c}_name'] = [channel_name]

            images_path = os.path.join(dst_path, f'Position_{p + 1}', 'Images')
            os.makedirs(images_path, exist_ok=True)

            video_data = None
            src_segm_paths = [''] * size_t
            frame_numbers = []
            size_z = 1
            for frame_i, file_info in enumerate(sorted_files_list):
                file, _ = file_info
                src_img_file_path = os.path.join(src_path, file)
                try:
                    img = load.imread(src_img_file_path)
                    if video_data is None:
                        video_data = np.zeros((size_t, *img.shape), dtype=img.dtype)
                    video_data[frame_i] = img
                    frame_number_match = re.findall(frame_name_pattern, file)[0][1]
                    frame_numbers.append(int(frame_number_match))
                except Exception:
                    continue

                if segm_folder and c == 0:
                    src_segm_paths[frame_i] = os.path.join(segm_folder, file)

                if img.ndim == 3:
                    size_z = len(img)
                df_metadata['SizeZ'] = [size_z]

            if video_data is None:
                logger_func(
                    f'WARNING: No valid image files found for position '
                    f'"{pos_name}", channel "{channel_name}"'
                )
                continue

            img_file_name = f'{pos_basename}{channel_name}.tif'
            dst_img_file_path = os.path.join(images_path, img_file_name)
            dst_segm_file_name = f'{pos_basename}segm_{channel_name}.npz'
            dst_segm_path = os.path.join(images_path, dst_segm_file_name)
            all_pos_data_info.append({
                'path': dst_img_file_path,
                'SizeT': size_t,
                'SizeZ': size_z,
                'data': video_data,
                'frameNumbers': frame_numbers,
                'dst_segm_path': dst_segm_path,
                'src_segm_paths': src_segm_paths,
            })

        metadata_csv_path = os.path.join(
            images_path, f'{pos_basename}metadata.csv'
        )
        df_metadata = df_metadata.T
        df_metadata.index.name = 'Description'
        df_metadata.to_csv(metadata_csv_path)
        logger_func('*' * 40)

    if not all_pos_data_info:
        raise RuntimeError('No valid image files found to restructure.')

    logger_func('Saving image files...')
    max_size_t = max(d['SizeT'] for d in all_pos_data_info)
    min_frame_number = min(d['frameNumbers'][0] for d in all_pos_data_info)
    for img_data_info in all_pos_data_info:
        video_data = img_data_info['data']
        frame_numbers = img_data_info['frameNumbers']
        padded_shape = (max_size_t, *video_data.shape[1:])
        padded_video_data = np.zeros(padded_shape, dtype=video_data.dtype)
        for frame_number, img in zip(frame_numbers, video_data):
            frame_i = frame_number - min_frame_number
            padded_video_data[frame_i] = img
        img_data_info['paddedShape'] = padded_shape
        img_data_info['data'] = None
        myutils.to_tiff(img_data_info['path'], padded_video_data)

    if not segm_folder:
        logger_func(f'Restructure completed. Output saved to "{dst_path}".')
        return

    logger_func('Saving segmentation files...')
    for img_data_info in all_pos_data_info:
        padded_shape = img_data_info['paddedShape']
        segm_data = np.zeros(padded_shape, dtype=np.uint32)
        for frame_number, segm_file_path in zip(
            img_data_info['frameNumbers'], img_data_info['src_segm_paths']
        ):
            frame_i = frame_number - min_frame_number
            try:
                lab = load.imread(segm_file_path).astype(np.uint32)
                segm_data[frame_i] = lab
            except Exception:
                logger_func(
                    'WARNING: Segmentation file does not exist, saving empty '
                    f'masks: "{segm_file_path}"'
                )
        io.savez_compressed(img_data_info['dst_segm_path'], segm_data)

    logger_func(f'Restructure completed. Output saved to "{dst_path}".')


def run_restructure_cli(args: dict, logger_func=print):
    src_path = os.path.abspath(args['input'])
    dst_path = os.path.abspath(args['output'])
    layout = args['layout']

    if not os.path.isdir(src_path):
        raise NotADirectoryError(
            f'Input path must be a folder containing image files: "{src_path}"'
        )
    os.makedirs(dst_path, exist_ok=True)

    if layout == 'multi-channel':
        restructure_multi_channel(
            src_path, dst_path,
            action=args.get('action', 'copy'),
            logger_func=logger_func,
        )
    elif layout == 'multi-timepoint':
        channels_arg = args.get('channels')
        if not channels_arg:
            raise ValueError(
                '--channels is required for layout "multi-timepoint".'
            )
        channels = [
            ch.strip() for ch in channels_arg.split(',') if ch.strip()
        ]
        restructure_multi_timepoint(
            src_path, dst_path,
            channels=channels,
            basename=args.get('basename') or '',
            segm_folder=args.get('segm_folder') or '',
            logger_func=logger_func,
        )
    else:
        valid = ', '.join(RESTRUCTURE_LAYOUTS)
        raise ValueError(
            f'Invalid restructure layout "{layout}". Valid values: {valid}'
        )


def _add_metadata_override_args(parser):
    parser.add_argument(
        '--trust-metadata', action=argparse.BooleanOptionalAction,
        default=True,
        help='Trust metadata read from the file (default: true for CLI).',
    )
    parser.add_argument('--lens-na', type=float, default=None)
    parser.add_argument('--size-t', type=int, default=None)
    parser.add_argument('--size-z', type=int, default=None)
    parser.add_argument('--size-c', type=int, default=None)
    parser.add_argument('--size-s', type=int, default=None)
    parser.add_argument('--time-increment', type=float, default=None)
    parser.add_argument('--physical-size-x', type=float, default=None)
    parser.add_argument('--physical-size-y', type=float, default=None)
    parser.add_argument('--physical-size-z', type=float, default=None)
    parser.add_argument(
        '--channels', type=str, default=None,
        help='Comma-separated channel names.',
    )
    parser.add_argument(
        '--em-wavelens', type=str, default=None,
        help='Comma-separated emission wavelengths.',
    )
    parser.add_argument(
        '--positions', type=str, default=None,
        help='Comma-separated positions to save (e.g. Position_1,Position_3). '
             'Default: all positions.',
    )
    parser.add_argument(
        '--time-range', type=str, default=None,
        help='Time range to save as start:end (0-indexed, inclusive).',
    )
    parser.add_argument(
        '--save-channels', type=str, default=None,
        help='Comma-separated booleans for channels to save (e.g. true,false,true).',
    )
    parser.add_argument(
        '--metadata-csv', type=str, default=None,
        help='CSV file with metadata overrides (Description,values format).',
    )
    parser.add_argument(
        '--add-image-name', action='store_true',
        help='Include image name in output filenames.',
    )


def build_data_parser():
    data_parser = argparse.ArgumentParser(
        prog='acdc-data',
        description='Headless data structure tools for Cell-ACDC (BioIO only).',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = data_parser.add_subparsers(dest='command', required=True)

    metadata_parser = subparsers.add_parser(
        'metadata',
        help='Extract metadata from a raw microscopy file.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    metadata_parser.add_argument(
        '--input', '-i', required=True, type=str,
        help='Path to a raw microscopy file or folder containing one file.',
    )
    metadata_parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output folder for metadata.csv and metadataXML.txt.',
    )
    metadata_parser.add_argument(
        '--format', choices=('text', 'json', 'csv'), default='text',
        help='Output format (default: text).',
    )
    metadata_parser.add_argument(
        '--basename', type=str, default=None,
        help='Basename for metadata.csv (default: derived from filename).',
    )

    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert raw microscopy files to the ACDC data structure.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    convert_parser.add_argument(
        '--input', '-i', required=True, type=str,
        help='Folder containing raw microscopy file(s).',
    )
    convert_parser.add_argument(
        '--output', '-o', required=True, type=str,
        help='Experiment destination folder.',
    )
    convert_parser.add_argument(
        '--layout', required=True,
        choices=tuple(LAYOUT_TO_RAW_DATA_STRUCT),
        help='How raw files are arranged.',
    )
    convert_parser.add_argument(
        '--format', choices=('tif', 'h5'), default='tif',
        help='Output image format (default: tif).',
    )
    convert_parser.add_argument(
        '--lazy-load', action=argparse.BooleanOptionalAction, default=True,
        help='Load one frame at a time to reduce RAM usage (default: true).',
    )
    convert_parser.add_argument(
        '--move-raw', action=argparse.BooleanOptionalAction, default=True,
        help='Move raw files to raw_microscopy_files/ after conversion.',
    )
    convert_parser.add_argument(
        '--on-existing', choices=('overwrite', 'add', 'create-new'),
        default='overwrite',
        help='Policy when destination already has Position folders.',
    )
    convert_parser.add_argument(
        '--basename', type=str, default=None,
        help='Required for layout one-per-channel.',
    )
    _add_metadata_override_args(convert_parser)

    restructure_parser = subparsers.add_parser(
        'restructure',
        help='Restructure pre-processed image files into the ACDC folder layout.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    restructure_parser.add_argument(
        '--input', '-i', required=True, type=str,
        help='Folder containing pre-processed image files.',
    )
    restructure_parser.add_argument(
        '--output', '-o', required=True, type=str,
        help='Destination experiment folder.',
    )
    restructure_parser.add_argument(
        '--layout', required=True, choices=RESTRUCTURE_LAYOUTS,
        help=(
            'How files are arranged: multi-timepoint (one file per frame, '
            'stack into TIFFs) or multi-channel (one file per channel, '
            'organize into Position folders).'
        ),
    )
    restructure_parser.add_argument(
        '--action', choices=('copy', 'move'), default='copy',
        help='For multi-channel layout: copy or move files (default: copy).',
    )
    restructure_parser.add_argument(
        '--channels', type=str, default=None,
        help=(
            'Required for multi-timepoint layout. Comma-separated channel '
            'names matching filenames (e.g. pos1_GFP_1.tif → GFP).'
        ),
    )
    restructure_parser.add_argument(
        '--basename', type=str, default=None,
        help='Optional basename prepended to output filenames (multi-timepoint).',
    )
    restructure_parser.add_argument(
        '--segm-folder', type=str, default=None,
        help=(
            'Optional folder with segmentation masks named like the raw files '
            '(multi-timepoint).'
        ),
    )

    return data_parser


def parse_data_cli_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    data_parser = build_data_parser()
    return vars(data_parser.parse_args(argv))


def run():
    from cellacdc import _run

    parser_args = parse_data_cli_args()
    command = parser_args['command']
    if command == 'metadata':
        _run.run_data_metadata(parser_args)
    elif command == 'convert':
        _run.run_data_convert(parser_args)
    elif command == 'restructure':
        _run.run_data_restructure(parser_args)
    else:
        raise ValueError(f'Unknown command "{command}"')


def main():
    run()
