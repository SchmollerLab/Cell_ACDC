"""C8: code run with the Python of a model environment (subprocess
templates) works with both layouts of cellacdc.myutils.

A model environment may hold another Cell-ACDC than the client (an
environment that is not repaired yet, or a user-managed one): myutils is a
package from C8 on, a single myutils.py before. Every template checks the
layout of the environment's Cell-ACDC with `hasattr(cellacdc.myutils,
"__path__")` and imports the new modules when it is a package, the old
single-module names otherwise (synthesis section 3; myutils shard note d).
No try/except ImportError, which would hide missing dependencies, and no
old import path on the new layout (no MovedWarning).

- myutils/model_installs.py::_available_gpu_frameworks_from_env (keeps the
  "_available_gpu_frameworks(True, False)" call text that
  tests/test_gpu_availability.py looks for);
- myutils/segm_models.py::getModelArgSpecFromEnv (myutils_models.py ->
  myutils/model_modules.py);
- service/environment.py: the new _MYUTILS_IMPORT_CODE imports every module
  of the myutils package (`import cellacdc.myutils` imported the whole old
  module) and binds `installs` to the module with the package spec helpers;
  _RUNTIME_CHECK_CODE, is_cellacdc_cli_installed and
  _check_environment_dependencies use it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

MODEL_INSTALLS = 'cellacdc/myutils/model_installs.py'
SEGM_MODELS = 'cellacdc/myutils/segm_models.py'
ENVIRONMENT = 'cellacdc/service/environment.py'

LAYOUT_NOTE = (
    "    # Runs with the Cell-ACDC of the model environment, which may still\n"
    "    # have the single-module myutils.py of older versions\n"
)


def main(repo):
    replace(repo, MODEL_INSTALLS,
            "def _available_gpu_frameworks_from_env(python_exec, cuda, directML):\n"
            "    script = f'''\n"
            "import json\n"
            "from cellacdc.myutils import _available_gpu_frameworks\n"
            "print(",
            "def _available_gpu_frameworks_from_env(python_exec, cuda, directML):\n"
            + LAYOUT_NOTE +
            "    script = f'''\n"
            "import json\n"
            "import cellacdc.myutils\n"
            "if hasattr(cellacdc.myutils, \"__path__\"):\n"
            "    from cellacdc.myutils.model_installs import _available_gpu_frameworks\n"
            "else:\n"
            "    from cellacdc.myutils import _available_gpu_frameworks\n"
            "print(",
            items=[MODEL_INSTALLS + '::_available_gpu_frameworks_from_env'])

    replace(repo, SEGM_MODELS,
            "    GUI types are transferred as metadata and reconstructed on the client.\n"
            "    \"\"\"\n"
            "    script = f'''\n"
            "import json, sys\n"
            "from cellacdc import myutils_models\n"
            "try:\n"
            "    importer = (\n"
            "        myutils_models.import_promptable_segment_module\n"
            "        if \"{model_kind}\" == \"promptable\"\n"
            "        else myutils_models.import_segment_module\n"
            "    )\n"
            "    acdcSegment = importer(\"{model_name}\")\n"
            "    init_params, segment_params = myutils_models.getModelArgSpec(acdcSegment)\n"
            "    url = getattr(acdcSegment, \"url_help\", lambda: None)()\n"
            "    def _serialize_p(p):\n"
            "        return [p.name, p.default, myutils_models._serialize_model_param_type(p.type), p.desc, p.docstring]\n",
            "    GUI types are transferred as metadata and reconstructed on the client.\n"
            "    \"\"\"\n"
            + LAYOUT_NOTE +
            "    script = f'''\n"
            "import json, sys\n"
            "import cellacdc.myutils\n"
            "if hasattr(cellacdc.myutils, \"__path__\"):\n"
            "    from cellacdc.myutils import model_modules\n"
            "else:\n"
            "    from cellacdc import myutils_models as model_modules\n"
            "try:\n"
            "    importer = (\n"
            "        model_modules.import_promptable_segment_module\n"
            "        if \"{model_kind}\" == \"promptable\"\n"
            "        else model_modules.import_segment_module\n"
            "    )\n"
            "    acdcSegment = importer(\"{model_name}\")\n"
            "    init_params, segment_params = model_modules.getModelArgSpec(acdcSegment)\n"
            "    url = getattr(acdcSegment, \"url_help\", lambda: None)()\n"
            "    def _serialize_p(p):\n"
            "        return [p.name, p.default, model_modules._serialize_model_param_type(p.type), p.desc, p.docstring]\n",
            items=[SEGM_MODELS + '::getModelArgSpecFromEnv'])

    replace(repo, ENVIRONMENT,
            "_RUNTIME_IMPORTS_OK = \"__ACDC_RUNTIME_IMPORTS_OK__\"\n"
            "_RUNTIME_CHECK_CODE = _RUNTIME_IDENTITY_CODE + f'''\n"
            "import cellacdc.core\n"
            "import cellacdc.myutils\n"
            "import cellacdc.service.worker\n"
            "print({_RUNTIME_IMPORTS_OK!r})\n"
            "'''\n",
            "_RUNTIME_IMPORTS_OK = \"__ACDC_RUNTIME_IMPORTS_OK__\"\n"
            "# Run in a model environment, whose Cell-ACDC may still have the\n"
            "# single-module myutils.py of older versions: imports every module of\n"
            "# the myutils package (as `import cellacdc.myutils` imported the whole\n"
            "# old module) and binds `installs` to the module with the package spec\n"
            "# helpers. No try/except ImportError: missing dependencies must fail\n"
            "_MYUTILS_IMPORT_CODE = '''\n"
            "import cellacdc.myutils\n"
            "if hasattr(cellacdc.myutils, \"__path__\"):\n"
            "    import importlib, pkgutil\n"
            "    for _acdc_module in pkgutil.iter_modules(cellacdc.myutils.__path__):\n"
            "        importlib.import_module(\"cellacdc.myutils.\" + _acdc_module.name)\n"
            "    from cellacdc.myutils import installs\n"
            "else:\n"
            "    from cellacdc import myutils as installs\n"
            "'''\n"
            "_RUNTIME_CHECK_CODE = _RUNTIME_IDENTITY_CODE + f'''\n"
            "import cellacdc.core\n"
            "{_MYUTILS_IMPORT_CODE}\n"
            "import cellacdc.service.worker\n"
            "print({_RUNTIME_IMPORTS_OK!r})\n"
            "'''\n",
            items=[ENVIRONMENT + '::_RUNTIME_CHECK_CODE',
                   ENVIRONMENT + '::_MYUTILS_IMPORT_CODE'])

    replace(repo, ENVIRONMENT,
            "    test_code = (\n"
            "        \"import cellacdc; \"\n"
            "        \"import cellacdc.core; \"\n"
            "        \"import cellacdc.myutils; \"\n"
            "        \"import cellacdc.service.worker\"\n"
            "    )\n",
            "    test_code = (\n"
            "        \"import cellacdc\\n\"\n"
            "        \"import cellacdc.core\\n\"\n"
            "        + _MYUTILS_IMPORT_CODE\n"
            "        + \"import cellacdc.service.worker\\n\"\n"
            "    )\n",
            items=[ENVIRONMENT + '::is_cellacdc_cli_installed'])

    replace(repo, ENVIRONMENT,
            "with startup_timer(\"Cell-ACDC runtime imports\"):\n"
            "    import cellacdc.core\n"
            "    from cellacdc import myutils\n"
            "    import cellacdc.service.worker\n",
            "with startup_timer(\"Cell-ACDC runtime imports\"):\n"
            "    import cellacdc.core\n"
            "    exec({_MYUTILS_IMPORT_CODE!r})\n"
            "    import cellacdc.service.worker\n",
            items=[ENVIRONMENT + '::_check_environment_dependencies'])
    replace(repo, ENVIRONMENT,
            "specs = myutils._normalize_pkg_specs(",
            "specs = installs._normalize_pkg_specs(",
            items=[ENVIRONMENT + '::_check_environment_dependencies'])
    replace(repo, ENVIRONMENT,
            "        myutils._check_package_version_constraints(spec, installed)\n",
            "        installs._check_package_version_constraints(spec, installed)\n",
            items=[ENVIRONMENT + '::_check_environment_dependencies'])


run(main)
