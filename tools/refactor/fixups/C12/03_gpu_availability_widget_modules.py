"""C12: test_directml_install_uses_target_python_and_version swapped the
global `widgets` of myutils.py for a SimpleNamespace, so that
_warn_install_gpu needs no Qt. _warn_install_gpu (myutils/model_installs.py)
reads the module globals widgets_message_box and widgets_push_buttons
(bound under `if GUI_INSTALLED:`): swap those two instead (raising=False:
they do not exist without Qt). The rewrite reports the old patch as
patch_unresolved and keeps the myutils import for it; it goes with it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

GPU = 'tests/test_gpu_availability.py'
TEST = GPU + '::test_directml_install_uses_target_python_and_version'


def main(repo):
    replace(repo, GPU,
            '''    monkeypatch.setattr(myutils, "widgets", SimpleNamespace(
        myMessageBox=lambda **kwargs: message,
        cancelPushButton=lambda text: stop,
        okPushButton=Mock(side_effect=[directml, proceed]),
    ), raising=False)
''',
            '''    # _warn_install_gpu reads the widget modules as globals of
    # model_installs (bound only when Qt is installed)
    monkeypatch.setattr(model_installs, "widgets_message_box", SimpleNamespace(
        myMessageBox=lambda **kwargs: message,
    ), raising=False)
    monkeypatch.setattr(model_installs, "widgets_push_buttons", SimpleNamespace(
        cancelPushButton=lambda text: stop,
        okPushButton=Mock(side_effect=[directml, proceed]),
    ), raising=False)
''', items=[TEST])
    replace(repo, GPU,
            "from cellacdc import myutils\n"
            "from cellacdc.myutils import installs, model_installs\n",
            "from cellacdc.myutils import installs, model_installs\n",
            items=[GPU + '::<imports>'])


run(main)
