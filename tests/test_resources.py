"""The Qt resources (icons, fonts, screenshots) are listed in
`cellacdc/resources_light.qrc` and `cellacdc/resources_dark.qrc` (same
aliases, dark files usually `<name>_dark.svg`) and compiled with pyrcc5 into
`cellacdc/qrc_resources_light.py` / `qrc_resources_dark.py`, which must be
recompiled whenever a qrc file or a listed file changes."""
import ast
import hashlib
import os
import re
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict

import pytest

import cellacdc

CELLACDC_PATH = os.path.dirname(os.path.abspath(cellacdc.__file__))
SCHEMES = ('light', 'dark')


def qrc_entries(scheme):
    qrc_path = os.path.join(CELLACDC_PATH, f'resources_{scheme}.qrc')
    with open(qrc_path, encoding='utf-8') as file:
        text = file.read()
    return re.findall(r'<file alias="([^"]+)">([^<]+)</file>', text)


def _bytes_literal(module_text, name):
    start = module_text.index(f'{name} = b"') + len(f'{name} = ')
    end = module_text.index('\n"', start) + 2
    return ast.literal_eval(module_text[start:end])


def compiled_resources(scheme):
    """``{name: bytes}`` of the compiled module, decoded without importing
    (i.e. registering) it. pyrcc5 format: `qt_resource_struct_v1` has a
    14-byte node per file/folder (name offset, flags, then child count and
    first child for folders, locale and data offset for files),
    `qt_resource_name` has per name a 2-byte length, a 4-byte hash and UTF-16
    characters, and `qt_resource_data` has per file a 4-byte size and the
    bytes (zlib-compressed after a 4-byte original size when flag 1 is set).
    """
    module_path = os.path.join(CELLACDC_PATH, f'qrc_resources_{scheme}.py')
    with open(module_path, encoding='utf-8') as file:
        text = file.read()
    data = _bytes_literal(text, 'qt_resource_data')
    names = _bytes_literal(text, 'qt_resource_name')
    tree = _bytes_literal(text, 'qt_resource_struct_v1')

    def number(buffer, start, size):
        return int.from_bytes(buffer[start:start+size], 'big')

    def name_at(offset):
        length = number(names, offset, 2)
        return names[offset+6:offset+6 + 2*length].decode('utf-16-be')

    resources = {}

    def add(node, folder):
        entry = tree[14*node:14*node + 14]
        flags = number(entry, 4, 2)
        name = name_at(number(entry, 0, 4)) if node else ''
        if flags & 2:
            first = number(entry, 10, 4)
            for child in range(first, first + number(entry, 6, 4)):
                add(child, f'{folder}{name}/' if node else '')
            return
        offset = number(entry, 10, 4)
        blob = data[offset+4:offset+4 + number(data, offset, 4)]
        assert not flags & 4, 'zstd compressed: compile with pyrcc5'
        if flags & 1:
            blob = zlib.decompress(blob[4:])
        resources[folder + name] = blob

    add(0, '')
    return resources


def test_light_and_dark_qrc_list_the_same_aliases():
    light, dark = (
        {alias for alias, _ in qrc_entries(scheme)} for scheme in SCHEMES
    )
    assert light - dark == set()
    assert dark - light == set()


@pytest.mark.parametrize('scheme', SCHEMES)
def test_qrc_files_exist_and_are_not_copies(scheme):
    files_by_content = defaultdict(set)
    for alias, rel_path in qrc_entries(scheme):
        filepath = os.path.join(CELLACDC_PATH, rel_path)
        assert os.path.isfile(filepath), (alias, rel_path)
        with open(filepath, 'rb') as file:
            content = file.read().replace(b'\r\n', b'\n')
        files_by_content[hashlib.md5(content).hexdigest()].add(rel_path)
    # Two files with the same content are usually an icon saved over another
    # one (e.g., a light icon left as a copy of the icon it was made from)
    copies = [paths for paths in files_by_content.values() if len(paths) > 1]
    assert copies == []


# Icons drawn differently by Qt 5 (see test below), to fix: remove the
# conflicting attribute (as for the measurements viewer icons), then drop the
# file from this set
QT5_COLOR_CONFLICTS_TO_FIX = {
    'resources/icons/clear_points_zoom_dark.svg',
    'resources/icons/compute-zoom_dark.svg',
    'resources/icons/continue.svg',
    'resources/icons/deselect_all.svg',
    'resources/icons/load_annotation.svg',
    'resources/icons/manual_background_dark.svg',
    'resources/icons/math/backspace.svg',
    'resources/icons/math/backspace_dark.svg',
    'resources/icons/select-list_dark.svg',
    'resources/icons/select_all.svg',
    'resources/icons/select_all_dark.svg',
    'resources/icons/terminal_down.svg',
    'resources/icons/terminal_up.svg',
    'resources/icons/whitelist_dark.svg',
    'resources/icons/zoom_rect_dark.svg',
}
STYLE_PROPERTIES = (
    'fill', 'stroke', 'opacity', 'fill-opacity', 'stroke-opacity',
    'stroke-width'
)


def svg_color_conflicts(filepath):
    """Elements with a presentation attribute (e.g. ``fill="#4d4d4d"``) that
    differs from the same property in their ``style``: Inkscape writes both,
    Qt 6 (and Inkscape) draw the style value, Qt 5 the attribute."""
    conflicts = []
    for element in ET.parse(filepath).iter():
        style = element.get('style')
        if not style:
            continue
        declarations = dict(
            (name.strip(), value.strip()) for name, value in (
                declaration.split(':', 1)
                for declaration in style.split(';') if ':' in declaration
            )
        )
        for name in STYLE_PROPERTIES:
            value = element.get(name)
            if value is None or name not in declarations:
                continue
            if value.strip().lower() != declarations[name].lower():
                conflicts.append((element.get('id'), name, value))
    return conflicts


@pytest.mark.parametrize('scheme', SCHEMES)
def test_svg_colors_do_not_depend_on_the_qt_version(scheme):
    conflicting = {}
    for alias, rel_path in set(qrc_entries(scheme)):
        if not rel_path.endswith('.svg'):
            continue
        conflicts = svg_color_conflicts(os.path.join(CELLACDC_PATH, rel_path))
        if conflicts and rel_path not in QT5_COLOR_CONFLICTS_TO_FIX:
            conflicting[rel_path] = conflicts
    assert conflicting == {}


@pytest.mark.parametrize('scheme', SCHEMES)
def test_compiled_resources_match_qrc(scheme):
    """Fails when the compiled module was not recompiled after changing the
    qrc file or a file it lists."""
    compiled = compiled_resources(scheme)
    expected = {}
    for alias, rel_path in qrc_entries(scheme):
        with open(os.path.join(CELLACDC_PATH, rel_path), 'rb') as file:
            expected[alias] = file.read()
    assert sorted(compiled) == sorted(expected)
    # Git checks text files (e.g. SVGs) out with CRLF on Windows (autocrlf),
    # so the compiled bytes depend on where the module was compiled
    outdated = [
        alias for alias in expected
        if compiled[alias].replace(b'\r\n', b'\n')
        != expected[alias].replace(b'\r\n', b'\n')
    ]
    assert outdated == []
