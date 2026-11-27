"""C13: test that ``python -m cellacdc.measurements_viewer [folder]`` (the
package's __main__.py, which replaced the ``if __name__ == '__main__'``
block of measurements_viewer.py) runs the viewer window's ``run``.

Appended to tests/test_launcher_measurements_viewer.py, after its last
test (exact anchor).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import _fixuplib as F  # noqa: E402

PATH = 'tests/test_launcher_measurements_viewer.py'
ANCHOR = '''    win.close()
    assert host.measurementsViewerWins == []
'''
TEST = '''

def test_python_m_measurements_viewer_runs_the_viewer(monkeypatch):
    """``python -m cellacdc.measurements_viewer [folder]`` (the package's
    ``__main__.py``) runs ``run`` of the viewer window module."""
    import runpy
    import sys
    calls = []
    monkeypatch.setattr(
        viewer_window, 'run', lambda folder=None: calls.append(folder)
    )
    monkeypatch.setattr(sys, 'argv', ['measurements_viewer', 'exp_folder'])
    runpy.run_module('cellacdc.measurements_viewer', run_name='__main__')
    monkeypatch.setattr(sys, 'argv', ['measurements_viewer'])
    runpy.run_module('cellacdc.measurements_viewer', run_name='__main__')
    assert calls == ['exp_folder', None]
'''


def main(repo):
    text, _ = F._read(os.path.join(repo, PATH))
    if 'def test_python_m_measurements_viewer_runs_the_viewer(' in text:
        # The new text contains the anchor: F.replace cannot tell
        print(f'  {PATH}: already done')
        return
    F.replace(repo, PATH, ANCHOR, ANCHOR + TEST,
              items=[f'{PATH}::test_python_m_measurements_viewer_runs_the_viewer'])


if __name__ == '__main__':
    F.run(main)
