"""C15: user and developer docs for the new package layout.

Exact replacements (fail loudly when the text changed on main):
- cellacdc/docs/source/model-environments.rst: myutils.* -> the myutils
  submodules (the plugin API keeps working as cellacdc.myutils.<name>), the
  Code structure table rows of myutils.py and segm.py/gui.py/workers.py,
  the GUI packages never imported in server mode, the location of the
  model parameter type (de)serializers (_serialize_model_param_type was in
  myutils_models.py, not myutils.py), the dialogs of QtHandler;
- cellacdc/docs/source/models.rst: check_install_package and the GUI
  packages a model must not import;
- cellacdc/docs/source/contributing.rst: a "Where the code lives" section
  pointing to CODEMAP.md;
- cellacdc/code_structure.md: the modules of the custom annotation code.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _fixuplib import replace, run  # noqa: E402

ENV = 'cellacdc/docs/source/model-environments.rst'
MODELS = 'cellacdc/docs/source/models.rst'
CONTRIB = 'cellacdc/docs/source/contributing.rst'
STRUCT = 'cellacdc/code_structure.md'


def model_environments(repo):
    replace(repo, ENV,
            "``cellacdc.RUNNING_AS_SERVER``. Use ``myutils.is_server_mode()`` to check it in\n"
            "code. In server mode:\n"
            "\n"
            "* Qt, ``cellacdc.widgets`` and ``cellacdc.apps`` are never imported. The model\n"
            "  environment does not even have Qt installed.\n"
            "* User configuration files are not read (an empty config is used).\n"
            "* Nothing is installed and nobody is asked anything.\n"
            "  ``myutils.check_install_package`` and ``check_install_torch`` raise a\n",
            "``cellacdc.RUNNING_AS_SERVER``. Use ``is_server_mode()`` of\n"
            "``cellacdc.myutils.installs`` to check it in code. In server mode:\n"
            "\n"
            "* Qt and the GUI packages (``cellacdc.gui``, ``cellacdc.shared.widgets``,\n"
            "  ``cellacdc.shared.dialogs``) are never imported. The model environment\n"
            "  does not even have Qt installed.\n"
            "* User configuration files are not read (an empty config is used).\n"
            "* Nothing is installed and nobody is asked anything.\n"
            "  ``check_install_package`` (``cellacdc.myutils.installs``) and\n"
            "  ``check_install_torch`` (``cellacdc.myutils.torch_utils``) raise a\n")
    replace(repo, ENV,
            "   * - ``cellacdc/myutils.py``\n"
            "     - ``getModelArgSpecFromEnv`` provides standalone parameter inspection in\n",
            "   * - ``cellacdc/myutils/segm_models.py``\n"
            "     - ``getModelArgSpecFromEnv`` provides standalone parameter inspection in\n")
    replace(repo, ENV,
            "   * - ``cellacdc/segm.py``, ``cellacdc/gui.py``, ``cellacdc/workers.py``\n"
            "     - GUI callers of ``ModelExecution``: the segmentation module\n"
            "       (``segmWin``), the main GUI (``repeatSegm``, also used by the magic\n"
            "       labeller, ``repeatSegmVideo`` and the YeastMate budding prediction\n"
            "       ``autoAssignBud_YeastMate``) and segmentation for lost IDs.\n",
            "   * - ``cellacdc/segmentation_module/main_window.py``,\n"
            "       ``cellacdc/gui/main_window/segmentation.py``,\n"
            "       ``cellacdc/gui/main_window/cca_annotation.py``,\n"
            "       ``cellacdc/gui/workers/segmentation_tracking.py``\n"
            "     - GUI callers of ``ModelExecution``: the segmentation module\n"
            "       (``segmWin``), the main GUI (``repeatSegm``, also used by the magic\n"
            "       labeller, ``repeatSegmVideo`` and the YeastMate budding prediction\n"
            "       ``autoAssignBud_YeastMate``) and segmentation for lost IDs.\n")
    replace(repo, ENV,
            "        # Packages, as pip names or as dicts with the arguments of\n"
            "        # myutils.check_install_package\n",
            "        # Packages, as pip names or as dicts with the arguments of\n"
            "        # check_install_package (cellacdc.myutils.installs)\n")
    replace(repo, ENV,
            "``__init__.py`` is imported in both execution paths. Use\n"
            "``myutils.check_install_package`` (it installs with confirmation in-process\n"
            "and raises a clear error in server mode) instead of calling ``pip`` yourself:\n"
            "\n"
            ".. code-block:: python\n"
            "\n"
            "    from cellacdc import myutils\n"
            "\n"
            "    myutils.check_install_package(\n"
            "        'mymodel', pypi_name='mymodel-seg', import_pkg_name='mymodel'\n"
            "    )\n",
            "``__init__.py`` is imported in both execution paths. Use\n"
            "``check_install_package`` of ``cellacdc.myutils.installs`` (it installs with\n"
            "confirmation in-process and raises a clear error in server mode) instead of\n"
            "calling ``pip`` yourself:\n"
            "\n"
            ".. code-block:: python\n"
            "\n"
            "    from cellacdc.myutils import installs\n"
            "\n"
            "    installs.check_install_package(\n"
            "        'mymodel', pypi_name='mymodel-seg', import_pkg_name='mymodel'\n"
            "    )\n"
            "\n"
            ".. note::\n"
            "\n"
            "    ``cellacdc.myutils`` was a single module before Cell-ACDC was\n"
            "    reorganized into packages (see ``CODEMAP.md`` in the repository). The\n"
            "    functions that models use (``check_install_package``,\n"
            "    ``is_server_mode``, ``get_model_path``, ``to_uint8``, ...) are still\n"
            "    available as ``myutils.<name>`` after ``from cellacdc import myutils``,\n"
            "    without a warning. A model kept outside this repository that must also\n"
            "    run with older Cell-ACDC versions can keep using that form. Models in\n"
            "    this repository import the submodule that defines the function.\n")
    replace(repo, ENV,
            "    except ModuleNotFoundError:\n"
            "        from cellacdc import myutils\n"
            "        if myutils.is_server_mode():\n"
            "            myutils._raise_server_missing('mymodel (MyModel setup_steps)')\n",
            "    except ModuleNotFoundError:\n"
            "        from cellacdc.myutils import installs\n"
            "        if installs.is_server_mode():\n"
            "            installs._raise_server_missing('mymodel (MyModel setup_steps)')\n")
    replace(repo, ENV,
            "Do not import Qt or ``cellacdc.widgets`` at module level in ``__init__.py`` or\n"
            "``acdcSegment.py``. They are not installed in the model environment.\n",
            "Do not import Qt or the GUI packages (``cellacdc.gui``,\n"
            "``cellacdc.shared.widgets``, ``cellacdc.shared.dialogs``) at module level in\n"
            "``__init__.py`` or ``acdcSegment.py``. Qt is not installed in the model\n"
            "environment.\n")
    replace(repo, ENV,
            "in ``__init__`` (see ``CsvFilePath``). Then register it in\n"
            "``_serialize_model_param_type`` and ``_deserialize_model_param_type`` in\n"
            "``myutils.py``. Unregistered widget types raise a ``TypeError``. The widget\n",
            "in ``__init__`` (see ``CsvFilePath``). Then register it in\n"
            "``_serialize_model_param_type`` (``cellacdc/myutils/model_modules.py``) and\n"
            "``_deserialize_model_param_type`` (``cellacdc/myutils/segm_models.py``).\n"
            "Unregistered widget types raise a ``TypeError``. The widget\n")
    replace(repo, ENV,
            "To send other data, load it on the client (see\n"
            "``myutils.resolve_segm_model_init_data``) and put the array or DataFrame into\n",
            "To send other data, load it on the client (see\n"
            "``resolve_segm_model_init_data`` in ``cellacdc/myutils/segm_models.py``) and\n"
            "put the array or DataFrame into\n")
    replace(repo, ENV,
            "    from cellacdc import myutils\n"
            "    from cellacdc.service.environment import get_or_create_model_env\n"
            "    from cellacdc.service.prompts import CliHandler\n"
            "\n"
            "    python = get_or_create_model_env('MyModel', prompt_func=CliHandler().ask)\n"
            "    init_params, segment_params, url = myutils.getModelArgSpecFromEnv(\n",
            "    from cellacdc.myutils import segm_models\n"
            "    from cellacdc.service.environment import get_or_create_model_env\n"
            "    from cellacdc.service.prompts import CliHandler\n"
            "\n"
            "    python = get_or_create_model_env('MyModel', prompt_func=CliHandler().ask)\n"
            "    init_params, segment_params, url = segm_models.getModelArgSpecFromEnv(\n")
    replace(repo, ENV,
            "       the standard Cell-ACDC dialogs: ``widgets.myMessageBox`` for\n"
            "       ``info``, ``confirm`` and ``choice`` with up to three options (one\n"
            "       button per option), ``apps.QDialogCombobox`` for longer choices,\n"
            "       ``apps.QLineEditDialog`` for ``text_input`` and\n"
            "       ``apps.InstallPyTorchDialog`` for ``dialog='pytorch'``.\n",
            "       the standard Cell-ACDC dialogs: ``myMessageBox``\n"
            "       (``cellacdc/shared/widgets/message_box.py``) for ``info``,\n"
            "       ``confirm`` and ``choice`` with up to three options (one button per\n"
            "       option), ``QDialogCombobox`` for longer choices and\n"
            "       ``QLineEditDialog`` for ``text_input``\n"
            "       (``cellacdc/shared/dialogs/generic.py``), and\n"
            "       ``InstallPyTorchDialog`` (``cellacdc/shared/dialogs/model_selection.py``)\n"
            "       for ``dialog='pytorch'``.\n")


def models(repo):
    replace(repo, MODELS,
            "   ``pip``. Use ``myutils.check_install_package('mymodel')``, which raises a\n"
            "   clear error inside the model environment if the configuration is\n"
            "   incomplete.\n",
            "   ``pip``. Use ``check_install_package('mymodel')`` of\n"
            "   ``cellacdc.myutils.installs``, which raises a clear error inside the model\n"
            "   environment if the configuration is incomplete.\n")
    replace(repo, MODELS,
            "4. Create ``acdcSegment.py`` with a ``Model`` class. Do not import Qt or\n"
            "   ``cellacdc.widgets`` in it:\n",
            "4. Create ``acdcSegment.py`` with a ``Model`` class. Do not import Qt or the\n"
            "   GUI packages (``cellacdc.gui``, ``cellacdc.shared.widgets``,\n"
            "   ``cellacdc.shared.dialogs``) in it:\n")


def contributing(repo):
    replace(repo, CONTRIB,
            "Note that if you want to modify the code of the Pull Request, you can simply \n"
            "commit and push to the same branch. GitHub will automatically add to the open \n"
            "Pull Request.\n"
            "\n"
            "Running the tests\n"
            "-----------------\n",
            "Note that if you want to modify the code of the Pull Request, you can simply \n"
            "commit and push to the same branch. GitHub will automatically add to the open \n"
            "Pull Request.\n"
            "\n"
            "Where the code lives\n"
            "--------------------\n"
            "\n"
            "The file ``CODEMAP.md`` at the root of the repository\n"
            "(`on GitHub <https://github.com/SchmollerLab/Cell_ACDC/blob/main/CODEMAP.md>`_)\n"
            "is the map of the ``cellacdc`` package: one line per module, where new code\n"
            "goes, which module to change for common tasks and where the names of the old\n"
            "single modules (``gui.py``, ``apps.py``, ``widgets.py``, ``myutils.py``, ...)\n"
            "went when Cell-ACDC was reorganized into packages. In short, each part of the\n"
            "program is a package with its own dialogs, widgets and workers\n"
            "(``launcher``, ``data_structure``, ``data_prep``, ``segmentation_module``,\n"
            "``gui``, ``measurements_viewer``, ``utils``), code used by several parts is\n"
            "in ``shared``, and the library modules without a window stay at the top level\n"
            "of ``cellacdc``. When you add or move a module, run\n"
            "``python tools/codemap.py`` and describe the new entry in ``CODEMAP.md``\n"
            "(``tests/test_codemap.py`` checks it).\n"
            "\n"
            "The old import paths (for example ``cellacdc.apps`` or\n"
            "``cellacdc.gui_navigation``) still work for code outside Cell-ACDC, with a\n"
            "``DeprecationWarning`` that names the new path. Cell-ACDC's own code and tests\n"
            "must use the new paths: the tests fail on the warning.\n"
            "\n"
            "Running the tests\n"
            "-----------------\n")


def code_structure(repo):
    replace(repo, STRUCT,
            "# Custom annotations in the main GUI\n"
            "\n"
            "Currently we are implementing",
            "# Custom annotations in the main GUI\n"
            "\n"
            "The code is in the mixin `CustomAnnotationsGuiElements` of "
            "`cellacdc/gui/mixins/custom_annotations.py`, which the measurements "
            "viewer also composes (its overrides are in "
            "`cellacdc/measurements_viewer/main_window/custom_annotations.py`), and "
            "the dialog in `cellacdc/gui/mixins/dialogs.py`.\n"
            "\n"
            "Currently we are implementing")
    replace(repo, STRUCT,
            "This function will open the dialog `apps.customAnnotationDialog`",
            "This function will open the dialog `customAnnotationDialog` "
            "(`gui/mixins/dialogs.py`)")
    replace(repo, STRUCT,
            "see `widgets.pgScatterSymbolsCombobox`)",
            "see `pgScatterSymbolsCombobox` in `shared/widgets/combo_boxes.py`)")
    replace(repo, STRUCT,
            "using the `widgets.macShortcutToWindows` function.",
            "using the `macShortcutToWindows` function of "
            "`shared/widgets/line_edits.py`.")
    replace(repo, STRUCT,
            "This function adds a `widgets.CustomAnnotToolButton` (custom `paintEvent`).",
            "This function adds a `CustomAnnotToolButton` (`gui/mixins/widgets.py`, "
            "custom `paintEvent`).")
    replace(repo, STRUCT,
            "saved as a json file to both `cellacdc/temp/custom_annotations.json` "
            "path (initialized as a global variable after the imports)",
            "saved as a json file to both `<settings folder>/custom_annotations.json` "
            "path (`custom_annot_path`, a module global of "
            "`gui/mixins/custom_annotations.py`)")
    replace(repo, STRUCT,
            "called in the case `elif isCustomAnnot:` of `gui_mousePressEventImg1`.",
            "called in the case `elif isCustomAnnot:` of `gui_mousePressEventImg1` "
            "(`gui/main_window/mouse_left_image.py`).")
    replace(repo, STRUCT,
            "3. At the end of `gui.loadingDataCompleted` or in "
            "`gui.next_pos`/`gui.prev_pos` we call",
            "3. At the end of `loadingDataCompleted` (`gui/main_window/loading.py`) "
            "or in `next_pos`/`prev_pos` (`gui/main_window/frame_navigation.py`) "
            "we call")


def main(repo):
    model_environments(repo)
    models(repo)
    contributing(repo)
    code_structure(repo)


if __name__ == '__main__':
    run(main)
