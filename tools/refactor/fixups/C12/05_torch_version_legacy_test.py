"""C12: test_legacy_imports_reexport_torch_helpers compared the torch
helpers of the myutils facade (plugin API, no warning) with the old module
_torch_utils (now an alias shim that warns on import, kept for the value
use getattr(_torch_utils, name), which the rewrite reports). Compare them
with myutils/torch_utils.py, where they live."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

TORCH = 'tests/test_torch_version.py'


def main(repo):
    replace(repo, TORCH,
            "from cellacdc import _torch_utils, myutils\n",
            "from cellacdc import myutils\n",
            items=[TORCH + '::<imports>'])
    replace(repo, TORCH,
            "        assert getattr(myutils, name) is getattr(_torch_utils, name)\n",
            "        assert getattr(myutils, name) is getattr(torch_utils, name)\n",
            items=[TORCH + '::test_legacy_imports_reexport_torch_helpers'])


run(main)
