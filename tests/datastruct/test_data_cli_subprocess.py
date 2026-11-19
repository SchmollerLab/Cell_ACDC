"""`acdc-data` in child interpreters, as the console script and as
`python -m cellacdc.data_cli`.

Four children, `PYTHONIOENCODING` and `PYTHONUTF8` unset (so standard
output is a cp1252 pipe on Windows) unless stated,
`QT_QPA_PLATFORM=offscreen`:

- console-script bootstrap (`sys.argv[0] = 'acdc-data'`) `convert --help`:
  acdc-data's own ASCII help, exit 0;
- the same bootstrap with `metadata -i x -p`: acdc-data's own usage error
  (exit 2, not the `acdc` parser's) and `config.parser_args['info']` stays
  False (`-i` is not taken by `cellacdc.config`);
- `python -m cellacdc.data_cli metadata --format json` on a fake raw file
  read in a worker child (test hook `ACDC_DATASTRUCT_READER_FACTORY`):
  the standard output bytes are exactly one ASCII JSON document, although
  the reader prints noise;
- `python -m cellacdc.data_cli metadata --format csv` with
  `PYTHONIOENCODING=cp1252` (on every platform): the CSV bytes are UTF-8,
  the encoding `convert --metadata-csv` reads, so a non-ASCII file stem and
  channel name round-trip.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from cellacdc.acdc_datastruct import metadata

from . import conditions, fake_bioio

pytestmark = conditions.subprocess_heavy

BOOTSTRAP = (
    "import sys; sys.argv[0] = 'acdc-data'; "
    "from cellacdc.data_cli import main; sys.exit(main())"
)


def child_env(**extra: str) -> dict[str, str]:
    env = conditions.hooks_env(**extra)
    env.pop('PYTHONIOENCODING', None)
    env.pop('PYTHONUTF8', None)
    return env


def run_child(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    """`python *args` with bytes output (timeout 180 s)."""
    return subprocess.run(
        [sys.executable, *args], env=env, cwd=conditions.REPO_ROOT,
        stdin=subprocess.DEVNULL, capture_output=True, timeout=180,
    )


def text(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')


def test_console_script_help():
    result = run_child('-c', BOOTSTRAP, 'convert', '--help', env=child_env())
    assert result.returncode == 0, text(result.stderr)
    assert result.stdout.isascii()
    assert result.stdout.decode('ascii').startswith(
        'usage: acdc-data convert'
    )
    assert b'--image-format' in result.stdout
    assert b'\r\r' not in result.stdout


def test_console_script_parses_its_own_arguments():
    code = (
        "import sys; sys.argv[0] = 'acdc-data'; "
        "from cellacdc import config; from cellacdc.data_cli import main; "
        "returncode = main(); "
        "sys.stdout.write(f\"INFO={config.parser_args['info']!r}\\n\"); "
        "sys.exit(returncode)"
    )
    result = run_child('-c', code, 'metadata', '-i', 'x', '-p',
                       env=child_env())
    err = text(result.stderr)
    assert result.returncode == 2, err
    # argparse reports unknown arguments with the top-level usage
    assert 'usage: acdc-data ' in err
    assert 'acdc-data: error: unrecognized arguments: -p' in err
    assert 'usage: acdc ' not in err
    assert result.stdout.decode('ascii').strip() == 'INFO=False'


def test_module_metadata_json_through_a_worker(tmp_path):
    raw = tmp_path / 'raw d\u00e0ta, \u00b5'
    raw.mkdir()
    fake_bioio.write_fake_raw(
        str(raw / 'pos1.czi'),
        fake_bioio.make_spec(channel_names=['GFP', 'mCherry'], noise='NOISE')
    )
    env = child_env(
        ACDC_DATASTRUCT_READER_FACTORY=fake_bioio.FAKE_BIOIMAGE_PATH
    )
    result = run_child(
        '-m', 'cellacdc.data_cli', 'metadata', '-i',
        os.path.join(str(raw), 'pos1.czi'), '--format', 'json',
        '--no-log-file', '-v', env=env
    )
    assert result.returncode == 0, text(result.stderr)
    data = json.loads(result.stdout.decode('ascii'))
    assert result.stdout == (json.dumps(data, indent=2) + '\n').encode()
    assert data['schema'] == 'acdc-data/metadata/1'
    assert data['file'] == os.path.join(str(raw), 'pos1.czi')
    assert data['reader'] == 'tests.datastruct.fake_bioio.FakeBioImage'
    assert data['metadata']['channel_names'] == ['GFP', 'mCherry']
    # the reader's print in the worker child reaches stderr only (-v)
    assert b'NOISE' in result.stderr and b'NOISE' not in result.stdout


def test_module_metadata_csv_round_trips_on_a_cp1252_stdout(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    path = os.path.join(str(raw), 'Zelle_\u00e4.czi')
    fake_bioio.write_fake_raw(
        path, fake_bioio.make_spec(channel_names=['GFP\u03bc', 'mCherry'])
    )
    env = child_env(
        ACDC_DATASTRUCT_READER_FACTORY=fake_bioio.FAKE_BIOIMAGE_PATH
    )
    env['PYTHONIOENCODING'] = 'cp1252'
    result = run_child(
        '-m', 'cellacdc.data_cli', 'metadata', '-i', path, '--format', 'csv',
        '--no-log-file', env=env
    )
    assert result.returncode == 0, text(result.stderr)
    out = result.stdout.decode('utf-8')
    assert 'basename,Zelle_\u00e4_\n' in out
    assert 'channel_0_name,GFP\u03bc\n' in out
    # what `convert --metadata-csv` reads from `metadata --format csv > file`
    csv_path = tmp_path / 'metadata.csv'
    csv_path.write_bytes(result.stdout)
    overrides, _ = metadata.overrides_from_metadata_csv(str(csv_path))
    assert overrides.channel_names == ('GFP\u03bc', 'mCherry')
