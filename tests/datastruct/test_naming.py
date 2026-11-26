"""acdc_datastruct.naming: sanitizing, validation, stems and basenames."""
from __future__ import annotations

import random

import pytest

from cellacdc.acdc_datastruct import naming


def _dialog_rule(chName):
    """Inline copy of QDialogMetadataXML.removeInvalidCharacters (main)."""
    chName = "".join(
        c if c.isalnum() or c=='_' or c=='' else '_' for c in chName
    )
    trim_ = chName.endswith('_')
    while trim_:
        chName = chName[:-1]
        trim_ = chName.endswith('_')
    return chName


def _random_strings(n: int, seed: int = 20261119):
    rng = random.Random(seed)
    alphabet = (
        'abcXYZ019_ -.!?/\\:;,()[]{}#@$%&*+=~\'"\t'
        'äöüßéèçñµμα中文١²Ⅰ'
        '\U0001f600​ '
    )
    out = ['', '_', '___', 'GFP', 'GFP_', 'GFP__', '_GFP', 'a b', '!']
    while len(out) < n:
        out.append(''.join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 12))
        ))
    return out


def test_sanitize_name_equals_dialog_rule():
    for text in _random_strings(2000):
        assert naming.sanitize_name(text) == _dialog_rule(text), repr(text)


@pytest.mark.parametrize('text, expected', [
    ('', ''),
    ('GFP', 'GFP'),
    ('GFP 1!', 'GFP_1'),
    ('mNeon-Green', 'mNeon_Green'),
    ('Alexa 488', 'Alexa_488'),
    ('GFP.ome', 'GFP_ome'),
    ('__', ''),
    ('_a_', '_a'),
    ('µm', 'µm'),
])
def test_sanitize_name_examples(text, expected):
    assert naming.sanitize_name(text) == expected


def test_validate_channel_names_ok():
    assert naming.validate_channel_names(['GFP', 'mCherry']) == []
    assert naming.validate_channel_names(['GFP', 'GFP'], [True, False]) == []
    assert naming.validate_channel_names(['', 'GFP'], [False, True]) == []
    assert naming.validate_channel_names([]) == []


def test_validate_channel_names_empty_and_casefold_duplicates():
    problems = naming.validate_channel_names(['', 'GFP'])
    assert len(problems) == 1 and 'Channel 0' in problems[0]
    assert 'empty' in problems[0]

    problems = naming.validate_channel_names(['GFP', 'gfp', 'RFP'])
    assert len(problems) == 1
    assert 'Channel 0' in problems[0] and 'channel 1' in problems[0]

    # 'GFP!' and 'GFP?' only collide after sanitizing
    names = [naming.sanitize_name(n) for n in ('GFP!', 'GFP?')]
    assert naming.validate_channel_names(names) != []

    # 'Straße' / 'STRASSE' are equal by casefold
    assert naming.validate_channel_names(['Straße', 'STRASSE']) != []


@pytest.mark.parametrize('filename, expected', [
    ('x_1_GFP.ome.tif', 'x_1_GFP'),
    ('x_1_GFP.OME.TIFF', 'x_1_GFP'),
    ('a.CZI', 'a'),
    ('a', 'a'),
    ('folder/b_2_RFP.tif', 'b_2_RFP'),
    ('exp.1_GFP.tif', 'exp.1_GFP'),
])
def test_raw_file_stem(filename, expected):
    assert naming.raw_file_stem(filename) == expected


@pytest.mark.parametrize('filename, expected', [
    ('x.ome.tif', 'x_ome'),
    ('exp1.czi', 'exp1'),
    ('ASY015_', 'ASY015_'),
    ('a.b.c.nd2', 'a_b_c'),
    ('noext', 'noext'),
])
def test_output_stem(filename, expected):
    assert naming.output_stem(filename) == expected


@pytest.mark.parametrize('number, num_positions, expected', [
    (1, 1, '1'),
    (1, 9, '1'),
    (1, 10, '01'),
    (12, 10, '12'),
    (3, 150, '003'),
    (1000, 150, '1000'),
])
def test_pos_str(number, num_positions, expected):
    assert naming.pos_str(number, num_positions) == expected


def test_position_basename_and_filename():
    assert naming.position_basename('exp1', '01') == 'exp1_s01_'
    assert naming.position_basename('ASY015_', '1') == 'ASY015__s1_'
    assert (
        naming.position_basename('x_ome', '2', 'Scene_1')
        == 'x_ome_Scene_1_s2_'
    )
    assert (
        naming.position_filename('exp1_s01_', 'metadata', '.csv')
        == 'exp1_s01_metadata.csv'
    )
    assert (
        naming.position_filename('exp1_s01_', 'metadataXML', '.txt')
        == 'exp1_s01_metadataXML.txt'
    )
