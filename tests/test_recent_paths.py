import pandas as pd
import pytest

from cellacdc import myutils


@pytest.mark.parametrize(
    'rows, expected_paths, expected_dates',
    [
        (
            [
                (r'C:\data\first', '2026-10-01'),
                ('C:/data/second', '2026-10-03'),
                ('C:/data/missing', '2026-10-05'),
                ('C:/data/first', '2026-10-04'),
                ('C:/data/second', '2026-10-02'),
            ],
            ['C:/data/first', 'C:/data/second'],
            ['2026-10-04', '2026-10-03'],
        ),
        (
            [(r'C:\data\first', '2026-10-01')],
            ['C:/data/first'],
            ['2026-10-01'],
        ),
        ([('C:/data/missing', '2026-10-01')], [], []),
        ([], [], []),
    ],
)
def test_recent_paths_dates_stay_aligned(
    tmp_path, monkeypatch, rows, expected_paths, expected_dates
):
    csv_path = tmp_path / 'recentPaths.csv'
    df = pd.DataFrame(rows, columns=['path', 'opened_last_on'])
    df.to_csv(csv_path, index_label='index')
    monkeypatch.setattr(myutils, 'recentPaths_path', str(csv_path))
    monkeypatch.setattr(
        myutils.os.path,
        'isdir',
        lambda path: path.replace('\\', '/') in {
            'C:/data/first', 'C:/data/second'
        },
    )

    assert myutils.get_recent_paths(None) == (expected_paths, expected_dates)


def test_recent_paths_without_dates(tmp_path, monkeypatch):
    csv_path = tmp_path / 'recentPaths.csv'
    existing_path = str(tmp_path)
    df = pd.DataFrame(
        {'path': [existing_path, str(tmp_path / 'missing'), existing_path]}
    )
    df.to_csv(csv_path, index_label='index')
    monkeypatch.setattr(myutils, 'recentPaths_path', str(csv_path))

    assert myutils.get_recent_paths(None) == (
        [existing_path.replace('\\', '/')], None
    )


def test_recent_paths_without_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        myutils, 'recentPaths_path', str(tmp_path / 'missing.csv')
    )

    assert myutils.get_recent_paths(None) == ([], None)
