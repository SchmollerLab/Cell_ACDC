"""Unit tests of tools/refactor/acdc_refactor/rewrite.py (phase 2 rewriter).

Run without the repository conftest:

    python -m pytest tools/refactor/tests/test_rewrite.py -q
        -p no:cacheprovider --rootdir tools/refactor --noconftest

The fake name table and the fake new-layout package used for the lookup
modules of patches live in rewrite_fixtures/.
"""
import importlib.util
import json
import os
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'rewrite_fixtures')
SRC_ROOT = os.path.join(FIXTURES, 'src')
TABLE_PATH = os.path.join(FIXTURES, 'name_table.json')
REWRITE_PATH = os.path.join(
    os.path.dirname(HERE), 'acdc_refactor', 'rewrite.py')


def _load_rewrite():
    name = 'acdc_refactor_rewrite_under_test'
    spec = importlib.util.spec_from_file_location(name, REWRITE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rw = _load_rewrite()


@pytest.fixture(scope='module')
def table():
    return rw.load_table(TABLE_PATH)


@pytest.fixture(scope='module')
def indexed_table():
    return rw.load_table(TABLE_PATH, index=rw.SourceIndex(SRC_ROOT))


def run(source, table, module='tests.test_x', is_package=False, **kwargs):
    source = textwrap.dedent(source).lstrip('\n')
    new, report = rw.rewrite_source(
        source, table, module_name=module, is_package=is_package, **kwargs)
    again, _ = rw.rewrite_source(
        new, table, module_name=module, is_package=is_package, **kwargs)
    assert again == new, 'not idempotent:\n' + again
    compile(new, '<rewritten>', 'exec')
    return new, report


def expect(text):
    return textwrap.dedent(text).lstrip('\n')


def kinds(report, key='unhandled'):
    return [entry['kind'] for entry in report[key]]


# -- module imports ----------------------------------------------------------

def test_module_import_in_a_test_file_becomes_absolute_aliases(table):
    new, report = run('''
        from cellacdc import apps


        def test_x():
            dlg = apps.QDialogCombobox(apps.SetMeasurementsDialog)
            return apps.QLineEditDialog
        ''', table)
    assert new == expect('''
        from cellacdc.shared.dialogs import (
            generic as dialogs_generic,
            set_measurements as dialogs_set_measurements,
        )


        def test_x():
            dlg = dialogs_generic.QDialogCombobox(dialogs_set_measurements.SetMeasurementsDialog)
            return dialogs_generic.QLineEditDialog
        ''')
    assert [r['new'] for r in report['references']] == [
        'dialogs_generic.QDialogCombobox',
        'dialogs_set_measurements.SetMeasurementsDialog',
        'dialogs_generic.QLineEditDialog',
    ]
    assert not report['unhandled']


def test_relative_inside_the_part_absolute_outside(table):
    new, _ = run('''
        from ... import apps


        def f():
            return apps.editCcaTableWidget, apps.QDialogCombobox
        ''', table, module='cellacdc.gui.mixins.overlay')
    assert new == expect('''
        from cellacdc.shared.dialogs import generic as dialogs_generic
        from ..dialogs import editing as gui_dialogs_editing


        def f():
            return gui_dialogs_editing.editCcaTableWidget, dialogs_generic.QDialogCombobox
        ''')


def test_local_alias_is_replaced_by_the_canonical_alias(table):
    new, _ = run('''
        from .. import widgets as acdc_widgets

        box = acdc_widgets.myMessageBox()
        ''', table, module='cellacdc.utils.align')
    assert new == expect('''
        from cellacdc.shared.widgets import message_box as widgets_message_box

        box = widgets_message_box.myMessageBox()
        ''')


def test_import_as_of_an_old_module(table):
    new, _ = run('''
        import cellacdc.apps as m

        m.QDialogCombobox()
        ''', table)
    assert new == expect('''
        from cellacdc.shared.dialogs import generic as dialogs_generic

        dialogs_generic.QDialogCombobox()
        ''')


def test_package_chains_keep_import_cellacdc_when_still_used(table):
    new, _ = run('''
        import cellacdc.widgets
        import os

        cellacdc.widgets.myMessageBox()
        cellacdc.printl(os.sep)
        ''', table)
    assert new == expect('''
        import cellacdc
        from cellacdc.shared.widgets import message_box as widgets_message_box
        import os

        widgets_message_box.myMessageBox()
        cellacdc.printl(os.sep)
        ''')


def test_package_chain_import_dropped_when_unused(table):
    new, _ = run('''
        import cellacdc.myutils

        cellacdc.myutils.check_install_package('x')
        ''', table)
    assert new == expect('''
        from cellacdc.myutils import installs

        installs.check_install_package('x')
        ''')


def test_chains_through_reexported_modules(table):
    new, _ = run('''
        from cellacdc import apps, widgets

        apps.widgets.myMessageBox.Yes
        widgets.apps.QLineEditDialog()
        ''', table)
    assert new == expect('''
        from cellacdc.shared.dialogs import generic as dialogs_generic
        from cellacdc.shared.widgets import message_box as widgets_message_box

        widgets_message_box.myMessageBox.Yes
        dialogs_generic.QLineEditDialog()
        ''')


def test_uses_in_nested_functions_classes_and_comprehensions(table):
    new, _ = run('''
        from cellacdc import widgets


        class A(widgets.myMessageBox):
            box = widgets.myMessageBox

            def f(self, cls=widgets.myMessageBox):
                def g():
                    return [widgets.myMessageBox for _ in range(2)]
                return lambda: widgets.myMessageBox
        ''', table)
    assert 'widgets.' not in new.replace('widgets_message_box.', '')
    assert new.count('widgets_message_box.myMessageBox') == 5


def test_shadowing_names_are_never_rewritten(table):
    new, report = run('''
        from cellacdc import widgets


        def param(widgets):
            return widgets.myMessageBox


        def local():
            widgets = object()
            return widgets.myMessageBox


        class Holder:
            widgets = None
            x = widgets.myMessageBox

            def method(self):
                return widgets.myMessageBox


        values = [widgets.myMessageBox for widgets in range(3)]
        ''', table)
    assert new == expect('''
        from cellacdc.shared.widgets import message_box as widgets_message_box


        def param(widgets):
            return widgets.myMessageBox


        def local():
            widgets = object()
            return widgets.myMessageBox


        class Holder:
            widgets = None
            x = widgets.myMessageBox

            def method(self):
                return widgets_message_box.myMessageBox


        values = [widgets.myMessageBox for widgets in range(3)]
        ''')
    assert not report['unhandled']


def test_alias_colliding_with_a_name_of_the_file_gets_mod_suffix(table):
    new, report = run('''
        from cellacdc import apps


        def f():
            dialogs_generic = 1
            return apps.QDialogCombobox, dialogs_generic
        ''', table)
    assert 'from cellacdc.shared.dialogs import generic as ' \
        'dialogs_generic_mod' in new
    assert 'return dialogs_generic_mod.QDialogCombobox, dialogs_generic' \
        in new
    assert kinds(report) == ['alias_collision']


def test_global_declaration_reaches_the_module_import(table):
    new, _ = run('''
        from cellacdc import apps


        def f():
            global apps
            return apps.QDialogCombobox
        ''', table)
    assert 'return dialogs_generic.QDialogCombobox' in new


# -- statement editing -------------------------------------------------------

def test_other_names_of_the_statement_are_kept(table):
    new, _ = run('''
        from .. import apps, load, widgets, html_utils  # noqa

        load.x(apps.QDialogCombobox, widgets.myMessageBox, html_utils)
        ''', table, module='cellacdc.utils.base')
    assert new == expect('''
        from .. import load, html_utils  # noqa
        from cellacdc.shared.dialogs import generic as dialogs_generic
        from cellacdc.shared.widgets import message_box as widgets_message_box

        load.x(dialogs_generic.QDialogCombobox, widgets_message_box.myMessageBox, html_utils)
        ''')


def test_parenthesized_import_keeps_layout_and_comments(table):
    new, _ = run('''
        from cellacdc import (
            load,  # data
            apps,
            html_utils,
        )

        load, html_utils, apps.QDialogCombobox
        ''', table)
    assert new == expect('''
        from cellacdc import (
            load,  # data
            html_utils,
        )
        from cellacdc.shared.dialogs import generic as dialogs_generic

        load, html_utils, dialogs_generic.QDialogCombobox
        ''')


def test_function_level_import_is_replaced_in_place(table):
    new, _ = run('''
        def f():
            from cellacdc import widgets
            return widgets.myMessageBox


        def g(widgets):
            return widgets.myMessageBox
        ''', table)
    assert new == expect('''
        def f():
            from cellacdc.shared.widgets import message_box as widgets_message_box
            return widgets_message_box.myMessageBox


        def g(widgets):
            return widgets.myMessageBox
        ''')


def test_no_function_import_shadowing_a_module_level_equivalent(table):
    new, _ = run('''
        from cellacdc.shared.widgets import message_box as widgets_message_box


        def f():
            first = widgets_message_box.myMessageBox
            from cellacdc import widgets
            return first, widgets.myMessageBox
        ''', table)
    assert new == expect('''
        from cellacdc.shared.widgets import message_box as widgets_message_box


        def f():
            first = widgets_message_box.myMessageBox
            return first, widgets_message_box.myMessageBox
        ''')


def test_repeated_import_in_the_same_block_is_merged(table):
    new, _ = run('''
        def f():
            from cellacdc import myutils
            myutils.check_install_package('a')
            from cellacdc import myutils
            myutils.check_install_package('b')
        ''', table)
    assert new == expect('''
        def f():
            from cellacdc.myutils import installs
            installs.check_install_package('a')
            installs.check_install_package('b')
        ''')


def test_imports_in_separate_branches_are_not_merged(table):
    new, _ = run('''
        if x:
            from cellacdc import apps
        else:
            from cellacdc import apps
        apps.QDialogCombobox
        ''', table)
    assert new.count(
        'from cellacdc.shared.dialogs import generic as dialogs_generic'
    ) == 2


def test_ambiguous_binding_is_left_unchanged(table):
    source = '''
        try:
            from cellacdc import apps
        except ImportError:
            apps = None
        apps.QDialogCombobox
        '''
    new, report = run(source, table)
    assert new == expect(source)
    assert kinds(report) == ['ambiguous_binding']


def test_import_sharing_its_line_is_left_unchanged(table):
    source = '''
        import os; from cellacdc import apps
        apps.QDialogCombobox
        '''
    new, report = run(source, table)
    assert new == expect(source)
    assert kinds(report) == ['ambiguous_binding']


def test_unused_dotted_import_is_kept_for_its_side_effects(table):
    source = '''
        import cellacdc.widgets  # noqa: F401
        '''
    new, report = run(source, table)
    assert new == expect(source)
    assert kinds(report) == ['unused_import']


def test_string_annotations_naming_an_old_alias_are_reported(table):
    new, report = run('''
        from cellacdc import widgets


        def f(box: 'widgets.myMessageBox') -> 'list[widgets.QAction]':
            return widgets.myMessageBox
        ''', table)
    assert "def f(box: 'widgets.myMessageBox') -> 'list[widgets.QAction]':" \
        in new
    assert kinds(report) == ['string_annotation'] * 2


def test_unused_imports(table):
    new, report = run('''
        from cellacdc import apps  # noqa: F401
        from cellacdc import gui_navigation  # noqa: F401
        ''', table)
    assert new == expect('''
        from cellacdc import apps  # noqa: F401
        from cellacdc.gui.mixins import navigation as gui_navigation  # noqa: F401
        ''')
    assert kinds(report) == ['unused_import']
    assert kinds(report, 'info') == ['unused_import_rewritten']


# -- from-imports of names ----------------------------------------------------

def test_from_import_of_names_is_split_per_home(table):
    new, report = run('''
        from cellacdc.apps import (
            QDialogCombobox, SetMeasurementsDialog as SMD, editCcaTableWidget
        )
        from cellacdc.widgets import QAction
        ''', table)
    assert new == expect('''
        from cellacdc.gui.dialogs.editing import editCcaTableWidget
        from cellacdc.shared.dialogs.generic import QDialogCombobox
        from cellacdc.shared.dialogs.set_measurements import SetMeasurementsDialog as SMD
        from qtpy.QtGui import QAction
        ''')
    assert len(report['imports']) == 4


def test_relative_from_import_of_names(table):
    new, _ = run('''
        from ...apps import QDialogCombobox, editCcaTableWidget
        ''', table, module='cellacdc.gui.mixins.overlay')
    assert new == expect('''
        from cellacdc.shared.dialogs.generic import QDialogCombobox
        from ..dialogs.editing import editCcaTableWidget
        ''')


def test_from_import_of_a_name_that_now_lives_in_this_module(table):
    new, report = run('''
        if TYPE_CHECKING:
            from cellacdc.apps import QDialogCombobox


        class QDialogCombobox:
            pass
        ''', table, module='cellacdc.shared.dialogs.generic')
    assert new == expect('''
        if TYPE_CHECKING:
            pass


        class QDialogCombobox:
            pass
        ''')
    assert kinds(report, 'info') == ['self_import_removed']


def test_from_import_unknown_name_is_kept_and_reported(table):
    source = '''
        from cellacdc.apps import imshow_tk
        '''
    new, report = run(source, table)
    assert new == expect(source)
    assert kinds(report) == ['unknown_name']


# -- whole-module moves --------------------------------------------------------

def test_whole_module_move_keeps_alias_and_moves_leaving_names(table):
    new, _ = run('''
        from cellacdc import gui_navigation

        class W(gui_navigation.LutGuiElements, gui_navigation.MovedThing):
            pass
        ''', table)
    assert new == expect('''
        from cellacdc.gui.mixins import (
            navigation as gui_navigation,
            object_info as gui_mixin_object_info,
        )

        class W(gui_navigation.LutGuiElements, gui_mixin_object_info.MovedThing):
            pass
        ''')


def test_from_import_of_a_whole_moved_module(table):
    new, _ = run('''
        from cellacdc.gui import guiWin, _font
        ''', table)
    assert new == expect('''
        from cellacdc.gui.main_window.constants import _font
        from cellacdc.gui.main_window.window import guiWin
        ''')


def test_value_use_of_a_whole_moved_module_is_rewritten_and_reported(table):
    new, report = run('''
        from cellacdc import _main

        path = _main.__file__
        ''', table)
    assert new == expect('''
        from cellacdc.launcher import main_window as launcher_window

        path = launcher_window.__file__
        ''')
    assert kinds(report, 'info') == ['value_use_moved']


def test_new_package_paths_are_not_old_paths(table):
    source = '''
        from cellacdc.gui import widgets as gui_widgets
        from cellacdc.gui.main_window import window as gui_window
        from cellacdc.myutils import installs
        import cellacdc.gui.main_window.window

        gui_widgets.x, gui_window.guiWin, installs.check_install_package
        cellacdc.gui.main_window.window.guiWin
        '''
    new, report = run(source, table)
    assert new == expect(source)
    assert not report['unhandled']


def test_old_reexport_of_a_name_that_is_also_a_new_package(table):
    # gui.py did "from . import widgets": the old cellacdc.widgets, not the
    # new package cellacdc.gui.widgets
    new, _ = run('''
        from cellacdc.gui import widgets

        widgets.myMessageBox()
        ''', table)
    assert new == expect('''
        from cellacdc.shared.widgets import message_box as widgets_message_box

        widgets_message_box.myMessageBox()
        ''')


# -- plan.py table keys: entries, module_values, shims, intermediate --------------

PLAN_TABLE = {
    'modules': {
        'cellacdc.gui': 'cellacdc.gui.main_window.window',
        'cellacdc.measurements_viewer':
            'cellacdc.measurements_viewer.main_window.window',
    },
    'names': {
        'cellacdc.apps': {
            'widgets': 'cellacdc.widgets', 'tqdm': 'tqdm',
            'Dlg': 'cellacdc.shared.dialogs.generic',
            'QDialogCombobox': 'cellacdc.shared.dialogs.generic',
        },
        'cellacdc.widgets': {
            'myMessageBox': 'cellacdc.shared.widgets.message_box'},
        'cellacdc.whitelist': {
            'WhitelistGUIElements': 'cellacdc.gui.main_window.whitelist_ids'},
        'cellacdc.gui': {'_font': 'cellacdc.gui.main_window.constants'},
    },
    'entries': {
        'cellacdc.apps': {
            'widgets': ['cellacdc.widgets', None],
            'tqdm': ['tqdm', 'tqdm'],
            'Dlg': ['cellacdc.shared.dialogs.generic', 'QDialogCombobox'],
            'QDialogCombobox': [
                'cellacdc.shared.dialogs.generic', 'QDialogCombobox'],
        },
    },
    'module_values': {'cellacdc.apps': {'widgets': 'cellacdc.widgets'}},
    'intermediate': {
        'cellacdc.gui.mixins.viewer_tools': {
            'SearchIDGuiElements': [
                'cellacdc.gui.mixins.object_info', 'SearchIDGuiElements'],
        },
    },
    'shims': {
        'cellacdc.apps': 'split', 'cellacdc.widgets': 'split',
        'cellacdc.whitelist': 'real', 'cellacdc.gui': 'package',
    },
    'aliases': {
        'cellacdc.shared.dialogs.generic': 'dialogs_generic',
        'cellacdc.shared.widgets.message_box': 'widgets_message_box',
        'cellacdc.gui.main_window.whitelist_ids': 'gui_whitelist_ids',
        'cellacdc.gui.main_window.window': 'gui_window',
        'cellacdc.gui.main_window.constants': 'gui_constants',
        'cellacdc.gui.mixins.viewer_tools': 'gui_viewer_tools',
        'cellacdc.gui.mixins.object_info': 'gui_mixin_object_info',
        'cellacdc.whitelist': 'whitelist',
    },
    'packages': [
        'cellacdc.gui', 'cellacdc.gui.mixins', 'cellacdc.measurements_viewer',
        'cellacdc.measurements_viewer.main_window',
    ],
}
PLAN_TABLE['aliases'].update({
    'cellacdc.measurements_viewer.widgets': 'viewer_widgets',
    'cellacdc.measurements_viewer.main_window.window': 'viewer_window',
})


@pytest.fixture(scope='module')
def plan_table():
    return rw.NameTable(PLAN_TABLE)


def test_entries_module_values_and_renamed_reexports(plan_table):
    new, report = run('''
        from cellacdc import apps
        from cellacdc.apps import Dlg

        for item in apps.tqdm(range(3)):
            apps.Dlg(), apps.widgets.myMessageBox, Dlg
        ''', plan_table)
    assert new == expect('''
        from cellacdc.shared.dialogs import generic as dialogs_generic
        from cellacdc.shared.widgets import message_box as widgets_message_box
        from tqdm import tqdm
        from cellacdc.shared.dialogs.generic import QDialogCombobox as Dlg

        for item in tqdm(range(3)):
            dialogs_generic.QDialogCombobox(), widgets_message_box.myMessageBox, Dlg
        ''')
    assert not report['unhandled']


def test_module_that_still_exists_keeps_its_import(plan_table):
    source = '''
        from cellacdc import whitelist
        from cellacdc.whitelist import Whitelist, WhitelistGUIElements


        def f(monkeypatch):
            monkeypatch.setattr(whitelist, 'Whitelist', None)
            return whitelist.Whitelist, whitelist.WhitelistGUIElements, whitelist
        '''
    new, report = run(source, plan_table)
    assert new == expect('''
        from cellacdc import whitelist
        from cellacdc.gui.main_window import whitelist_ids as gui_whitelist_ids
        from cellacdc.whitelist import Whitelist
        from cellacdc.gui.main_window.whitelist_ids import WhitelistGUIElements


        def f(monkeypatch):
            monkeypatch.setattr(whitelist, 'Whitelist', None)
            return whitelist.Whitelist, gui_whitelist_ids.WhitelistGUIElements, whitelist
        ''')
    assert not report['unhandled']
    unchanged = '''
        from cellacdc import whitelist

        whitelist.Whitelist
        '''
    assert run(unchanged, plan_table)[0] == expect(unchanged)


def test_import_of_a_module_that_still_exists_left_unused_is_dropped(
        plan_table):
    source = '''
        from cellacdc import whitelist


        class W(whitelist.WhitelistGUIElements):
            pass
        '''
    new, report = run(source, plan_table, module='cellacdc.thing',
                      drop_unused=True, external_uses={})
    assert new == expect('''
        from cellacdc.gui.main_window import whitelist_ids as gui_whitelist_ids


        class W(gui_whitelist_ids.WhitelistGUIElements):
            pass
        ''')
    assert [(d['import'], d['local']) for d in report['dropped']] == [
        ('cellacdc.whitelist', 'whitelist')]
    # without the option, and when a use is left, the import stays
    new, _ = run(source, plan_table, module='cellacdc.thing')
    assert 'from cellacdc import whitelist\n' in new
    new, report = run(source + '    whitelist.Whitelist\n', plan_table,
                      module='cellacdc.thing', drop_unused=True,
                      external_uses={})
    assert 'from cellacdc import whitelist\n' in new
    assert report['dropped'] == []
    # an import without any use was not left unused by the rewrite
    unused = '''
        from cellacdc import whitelist
        '''
    new, report = run(unused, plan_table, module='cellacdc.thing',
                      drop_unused=True, external_uses={})
    assert new == expect(unused)
    assert report['dropped'] == []


def test_intermediate_module_keeps_the_names_it_still_defines(plan_table):
    new, _ = run('''
        from cellacdc.gui.mixins import viewer_tools as gui_viewer_tools


        class W(gui_viewer_tools.SearchIDGuiElements,
                gui_viewer_tools.ImageViewerGuiElements):
            pass
        ''', plan_table)
    assert new == expect('''
        from cellacdc.gui.mixins import viewer_tools as gui_viewer_tools
        from cellacdc.gui.mixins import object_info as gui_mixin_object_info


        class W(gui_mixin_object_info.SearchIDGuiElements,
                gui_viewer_tools.ImageViewerGuiElements):
            pass
        ''')


def test_value_use_of_a_module_replaced_by_a_package_is_flagged(plan_table):
    new, report = run('''
        import inspect
        from cellacdc import gui

        source = inspect.getsource(gui)
        ''', plan_table)
    assert 'source = inspect.getsource(gui_window)' in new
    assert kinds(report) == ['value_use_moved']


def test_reexport_omitted_because_it_clashes_with_a_new_submodule(
        plan_table):
    # measurements_viewer.py did "from . import widgets": the table omits
    # the name because cellacdc.measurements_viewer.widgets is now a module
    new, report = run('''
        import cellacdc.measurements_viewer.widgets
        from cellacdc import measurements_viewer


        def test_x(monkeypatch, box):
            monkeypatch.setattr(measurements_viewer.widgets, 'myMessageBox', box)
            cellacdc.measurements_viewer.widgets.ViewerWidget()
            return measurements_viewer.MeasurementsViewerWin
        ''', plan_table)
    assert new == expect('''
        import cellacdc.measurements_viewer.widgets
        from cellacdc.measurements_viewer.main_window import window as viewer_window
        from cellacdc.shared.widgets import message_box as widgets_message_box


        def test_x(monkeypatch, box):
            monkeypatch.setattr(widgets_message_box, 'myMessageBox', box)
            cellacdc.measurements_viewer.widgets.ViewerWidget()
            return viewer_window.MeasurementsViewerWin
        ''')
    assert kinds(report) == ['reexport_guessed']


# -- values and unknown names ----------------------------------------------------

def test_split_module_used_as_value_keeps_the_import(table):
    new, report = run('''
        from cellacdc import widgets


        def f(name):
            box = widgets.myMessageBox
            return getattr(widgets, name), widgets.__file__, print(widgets)
        ''', table)
    assert new == expect('''
        from cellacdc import widgets
        from cellacdc.shared.widgets import message_box as widgets_message_box


        def f(name):
            box = widgets_message_box.myMessageBox
            return getattr(widgets, name), widgets.__file__, print(widgets)
        ''')
    assert kinds(report) == ['value_use'] * 3


def test_unknown_name_is_reported(table):
    new, report = run('''
        from cellacdc import apps

        apps.imshow_tk, apps.QDialogCombobox
        ''', table)
    assert 'apps.imshow_tk, dialogs_generic.QDialogCombobox' in new
    assert 'from cellacdc import apps\n' in new
    assert kinds(report) == ['unknown_name']


def test_getattr_with_a_constant_name(table):
    new, _ = run('''
        from cellacdc import apps

        cls = getattr(apps, 'QDialogCombobox')
        ''', table)
    assert "cls = getattr(dialogs_generic, 'QDialogCombobox')" in new


# -- external homes ----------------------------------------------------------------

def test_name_with_external_home_is_imported_by_name(table):
    new, _ = run('''
        from cellacdc import widgets

        action = widgets.QAction()
        ''', table)
    assert new == expect('''
        from qtpy.QtGui import QAction

        action = QAction()
        ''')


def test_external_name_colliding_with_a_local_uses_a_module_alias(table):
    new, report = run('''
        from cellacdc import widgets


        def f():
            QAction = None
            return widgets.QAction, QAction
        ''', table)
    assert new == expect('''
        from qtpy import QtGui


        def f():
            QAction = None
            return QtGui.QAction, QAction
        ''')
    assert kinds(report) == ['alias_collision']


def test_external_origins_option(indexed_table):
    new, report = run('''
        from cellacdc import widgets

        widgets.QMenu, widgets.qtpy.compat, widgets.is_mac
        ''', indexed_table, external_origins=True)
    assert new == expect('''
        import qtpy
        from cellacdc.shared.widgets import line_edits as widgets_line_edits
        from qtpy.QtWidgets import QMenu

        QMenu, qtpy.compat, widgets_line_edits.is_mac
        ''')
    assert kinds(report, 'info') == ['external_origin'] * 2


def test_reference_to_a_name_now_defined_in_this_module(table):
    new, report = run('''
        from ... import apps


        def f():
            return apps.QDialogCombobox
        ''', table, module='cellacdc.shared.dialogs.generic')
    # the import line is removed, the blank lines after it stay
    assert new == 2 * '\n' + expect('''
        def f():
            return QDialogCombobox
        ''')
    assert kinds(report, 'info') == ['self_reference']


# -- patches ---------------------------------------------------------------------------

def test_patch_object_forms(table):
    new, report = run('''
        from unittest import mock
        from cellacdc import apps, widgets


        def test_x(monkeypatch):
            monkeypatch.setattr(widgets, 'myMessageBox', None)
            monkeypatch.delattr(widgets, 'is_mac', raising=False)
            setattr(apps, 'QDialogCombobox', None)
            with mock.patch.object(apps, 'SetMeasurementsDialog'):
                pass
            monkeypatch.setattr(apps.widgets, 'QDialogListbox', None)
            monkeypatch.setattr(apps.QDialogCombobox, 'exec_', None)
        ''', table)
    assert new == expect('''
        from unittest import mock
        from cellacdc.shared.dialogs import (
            generic as dialogs_generic,
            set_measurements as dialogs_set_measurements,
        )
        from cellacdc.shared.widgets import (
            item_views as widgets_item_views,
            line_edits as widgets_line_edits,
            message_box as widgets_message_box,
        )


        def test_x(monkeypatch):
            monkeypatch.setattr(widgets_message_box, 'myMessageBox', None)
            monkeypatch.delattr(widgets_line_edits, 'is_mac', raising=False)
            setattr(dialogs_generic, 'QDialogCombobox', None)
            with mock.patch.object(dialogs_set_measurements, 'SetMeasurementsDialog'):
                pass
            monkeypatch.setattr(widgets_item_views, 'QDialogListbox', None)
            monkeypatch.setattr(dialogs_generic.QDialogCombobox, 'exec_', None)
        ''')
    targets = [p['target'] for p in report['patches']]
    assert targets == [
        'cellacdc.widgets.myMessageBox', 'cellacdc.widgets.is_mac',
        'cellacdc.apps.QDialogCombobox', 'cellacdc.apps.SetMeasurementsDialog',
        'cellacdc.widgets.QDialogListbox',
    ]


def test_patch_string_forms(table):
    new, report = run('''
        from unittest.mock import patch


        @patch('cellacdc.apps.QDialogCombobox')
        def test_x(monkeypatch, mocked):
            monkeypatch.setattr('cellacdc.widgets.myMessageBox', None)
            monkeypatch.delattr('cellacdc.widgets.is_mac')
            with patch('cellacdc.apps.QDialogCombobox.exec_'):
                pass
        ''', table)
    assert new == expect('''
        from unittest.mock import patch


        @patch('cellacdc.shared.dialogs.generic.QDialogCombobox')
        def test_x(monkeypatch, mocked):
            monkeypatch.setattr('cellacdc.shared.widgets.message_box.myMessageBox', None)
            monkeypatch.delattr('cellacdc.shared.widgets.line_edits.is_mac')
            with patch('cellacdc.shared.dialogs.generic.QDialogCombobox.exec_'):
                pass
        ''')
    assert [p['form'] for p in report['patches']] == ['string'] * 4


def test_assignment_to_a_module_attribute_is_a_patch(table):
    new, report = run('''
        from cellacdc import widgets

        widgets.myMessageBox = object
        ''', table)
    assert 'widgets_message_box.myMessageBox = object' in new
    assert report['patches'][0]['form'] == 'assign'


def test_patch_of_a_reexported_name_targets_its_lookup_modules(
        indexed_table):
    new, report = run('''
        from unittest import mock
        from cellacdc import myutils


        def test_x(monkeypatch):
            monkeypatch.setattr(myutils, 'printl', print)
            monkeypatch.setattr(myutils, 'printl', mock.Mock())
            monkeypatch.setattr('cellacdc.myutils.printl', print)
            monkeypatch.setattr(myutils, 'get_torch_device', print)
        ''', indexed_table)
    assert new == expect('''
        from unittest import mock
        from cellacdc.myutils import filesystem, installs, torch_utils


        def test_x(monkeypatch):
            # moved: printl is read in each of these modules
            monkeypatch.setattr(filesystem, 'printl', print)
            monkeypatch.setattr(installs, 'printl', print)
            # moved: printl is read in each of these modules
            monkeypatch.setattr(filesystem, 'printl', mock.Mock())
            monkeypatch.setattr(installs, 'printl', filesystem.printl)
            monkeypatch.setattr('cellacdc.myutils.filesystem.printl', print)
            monkeypatch.setattr(torch_utils, 'get_torch_device', print)
        ''')
    # a string target cannot be repeated: only its first module is patched
    assert kinds(report) == ['patch_multiple_lookups']
    assert report['patches'][0]['new'] == [
        'cellacdc.myutils.filesystem.printl',
        'cellacdc.myutils.installs.printl',
    ]


def test_patch_of_module_globals_of_a_whole_moved_module(indexed_table):
    new, report = run('''
        from cellacdc import gui


        def test_x(monkeypatch, menu):
            monkeypatch.setattr(gui, '_font', None)
            monkeypatch.setattr(gui, 'QMenu', menu)
            return gui.guiWin
        ''', indexed_table)
    assert new == expect('''
        from cellacdc.gui.main_window import (
            constants as gui_constants,
            toolbars as gui_toolbars,
            window as gui_window,
        )


        def test_x(monkeypatch, menu):
            monkeypatch.setattr(gui_constants, '_font', None)
            # moved: QMenu is read in each of these modules
            monkeypatch.setattr(gui_toolbars, 'QMenu', menu)
            monkeypatch.setattr(gui_window, 'QMenu', menu)
            return gui_window.guiWin
        ''')
    assert not report['unhandled']


def test_patch_of_a_reexported_name_needs_the_source_index(table):
    source = '''
        from cellacdc import myutils


        def test_x(monkeypatch):
            monkeypatch.setattr(myutils, 'printl', print)
            monkeypatch.setattr(myutils, 'check_install_package', print)
        '''
    new, report = run(source, table)
    assert "monkeypatch.setattr(myutils, 'printl', print)" in new
    assert (
        "monkeypatch.setattr(installs, 'check_install_package', print)"
        in new)
    assert kinds(report) == ['patch_unresolved']
    assert '--src-root' in report['unhandled'][0]['detail']


def test_command_line_with_a_source_root(tmp_path):
    path = _write(
        tmp_path / 'test_a.py',
        b'from cellacdc import myutils\n\n\n'
        b'def test_x(monkeypatch):\n'
        b'    monkeypatch.setattr(myutils, "get_torch_device", None)\n')
    report_path = tmp_path / 'report.json'
    assert rw.main([
        '--table', TABLE_PATH, '--src-root', SRC_ROOT, '--report',
        str(report_path), str(path)]) == 0
    assert path.read_bytes() == (
        b'from cellacdc.myutils import torch_utils\n\n\n'
        b'def test_x(monkeypatch):\n'
        b'    monkeypatch.setattr(torch_utils, "get_torch_device", None)\n')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['src_root'] == os.path.abspath(SRC_ROOT)


def test_patch_of_a_reexported_module_attribute(table):
    new, _ = run('''
        from cellacdc import myutils


        def test_x(monkeypatch, run):
            monkeypatch.setattr(myutils.subprocess, 'run', run)
        ''', table)
    assert "monkeypatch.setattr(installs.subprocess, 'run', run)" in new


def test_dynamic_patch_is_reported(table):
    new, report = run('''
        from cellacdc import widgets


        def test_x(monkeypatch, name):
            monkeypatch.setattr(widgets, name, None)
        ''', table)
    assert 'monkeypatch.setattr(widgets, name, None)' in new
    assert kinds(report) == ['dynamic_patch']


# -- strings ---------------------------------------------------------------------------

def test_dotted_path_strings(table):
    new, report = run('''
        """Docstring naming cellacdc.apps.QDialogCombobox and gui.py."""
        import importlib

        a = importlib.import_module('cellacdc.gui_navigation')
        b = 'cellacdc.apps.QDialogCombobox'
        c = importlib.import_module('cellacdc.apps')
        d = 'cellacdc.measurements_viewer'
        e = "python -c 'import cellacdc.apps'"
        f = f'{a} cellacdc.widgets.myMessageBox'
        g = 'cellacdc.load.foo'
        ''', table)
    assert new == expect('''
        """Docstring naming cellacdc.apps.QDialogCombobox and gui.py."""
        import importlib

        a = importlib.import_module('cellacdc.gui.mixins.navigation')
        b = 'cellacdc.shared.dialogs.generic.QDialogCombobox'
        c = importlib.import_module('cellacdc.apps')
        d = 'cellacdc.measurements_viewer'
        e = "python -c 'import cellacdc.apps'"
        f = f'{a} cellacdc.widgets.myMessageBox'
        g = 'cellacdc.load.foo'
        ''')
    assert kinds(report) == ['string_old_path', 'string_ambiguous']
    assert [(m['kind'], m['text']) for m in report['mentions']] == [
        ('docstring', 'cellacdc.apps.QDialogCombobox'),
        ('docstring', 'gui.py'),
        ('string', 'cellacdc.apps'),
        ('fstring', 'cellacdc.widgets.myMessageBox'),
    ]
    assert len(report['strings']) == 2


@pytest.mark.skipif(not rw.FSTRING_POSITIONS_OK,
                    reason='f-string positions need Python >= 3.12')
def test_references_inside_fstrings(table):
    new, _ = run('''
        from cellacdc import apps

        text = f"{apps.QDialogCombobox.__name__!r:>10} é {apps.QLineEditDialog}"
        ''', table)
    assert (
        'f"{dialogs_generic.QDialogCombobox.__name__!r:>10} é '
        '{dialogs_generic.QLineEditDialog}"') in new


def test_long_lines_are_reported_not_wrapped(table):
    long_args = ', '.join('argument_%d' % i for i in range(11))
    new, report = run('''
        from cellacdc import widgets

        widgets.myMessageBox(%s)
        ''' % long_args, table)
    assert 'widgets_message_box.myMessageBox(%s)' % long_args in new
    assert [entry['line'] for entry in report['long_lines']] == [3]


# -- files, encodings and the command line ---------------------------------------------------

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_crlf_encoding_and_bom_are_preserved(tmp_path, table):
    source = (
        '# -*- coding: latin-1 -*-\r\n'
        'from cellacdc import widgets\r\n'
        '\r\n'
        "TEXT = 'café'\r\n"
        'box = widgets.myMessageBox()\r\n'
    )
    path = _write(tmp_path / 'crlf.py', source.encode('latin-1'))
    changed, _ = rw.rewrite_file(str(path), table)
    assert changed
    data = path.read_bytes()
    assert data == (
        '# -*- coding: latin-1 -*-\r\n'
        'from cellacdc.shared.widgets import message_box as '
        'widgets_message_box\r\n'
        '\r\n'
        "TEXT = 'café'\r\n"
        'box = widgets_message_box.myMessageBox()\r\n'
    ).encode('latin-1')
    bom = _write(
        tmp_path / 'bom.py',
        b'\xef\xbb\xbffrom cellacdc import widgets\r\nwidgets.is_mac\r\n')
    assert rw.rewrite_file(str(bom), table)[0]
    assert bom.read_bytes() == (
        b'\xef\xbb\xbffrom cellacdc.shared.widgets import line_edits as '
        b'widgets_line_edits\r\nwidgets_line_edits.is_mac\r\n')


def test_unchanged_files_are_not_written(tmp_path, table):
    path = _write(tmp_path / 'same.py', b'import os\r\nos.sep\r\n')
    os.utime(path, (1000000000, 1000000000))
    changed, _ = rw.rewrite_file(str(path), table)
    assert not changed
    assert os.stat(path).st_mtime == 1000000000


def test_syntax_errors_are_reported_and_skipped(tmp_path, table):
    data = b'from cellacdc import apps\napps.QDialogCombobox(\n'
    path = _write(tmp_path / 'broken.py', data)
    changed, report = rw.rewrite_file(str(path), table)
    assert not changed
    assert path.read_bytes() == data
    assert kinds(report) == ['syntax_error']


def test_module_names_from_init_files(tmp_path):
    _write(tmp_path / 'cellacdc' / '__init__.py', b'')
    _write(tmp_path / 'cellacdc' / 'gui' / '__init__.py', b'')
    module = _write(tmp_path / 'cellacdc' / 'gui' / 'window.py', b'')
    script = _write(tmp_path / 'script.py', b'')
    assert rw.module_name_for_path(str(module)) == (
        'cellacdc.gui.window', False)
    assert rw.module_name_for_path(
        str(tmp_path / 'cellacdc' / 'gui' / '__init__.py')) == (
        'cellacdc.gui', True)
    assert rw.module_name_for_path(str(script)) == (None, False)


def test_relative_import_from_a_file_in_the_package(tmp_path, table):
    _write(tmp_path / 'cellacdc' / '__init__.py', b'')
    _write(tmp_path / 'cellacdc' / 'utils' / '__init__.py', b'')
    path = _write(
        tmp_path / 'cellacdc' / 'utils' / 'align.py',
        b'from .. import widgets\nwidgets.myMessageBox()\n')
    assert rw.rewrite_file(str(path), table)[0]
    assert path.read_bytes() == (
        b'from cellacdc.shared.widgets import message_box as '
        b'widgets_message_box\nwidgets_message_box.myMessageBox()\n')


def test_hidden_test_files_and_default_excludes(tmp_path):
    tests = tmp_path / 'tests'
    hidden = _write(tests / 'segm' / '.test_sam.py', b'')
    plain = _write(tests / 'test_a.py', b'')
    _write(tests / 'test_related_tests.py', b'')
    _write(tests / 'test_compat_layer.py', b'')
    _write(tests / '__pycache__' / 'x.py', b'')
    _write(tests / '.hidden_dir' / 'test_b.py', b'')
    found = list(rw.iter_python_files([str(tests)], rw.DEFAULT_EXCLUDES))
    assert sorted(found) == sorted([str(hidden), str(plain)])
    assert list(rw.iter_python_files([str(hidden)], rw.DEFAULT_EXCLUDES)) \
        == [str(hidden)]


def test_command_line_report_and_check(tmp_path, capsys):
    tests = tmp_path / 'tests'
    changed = _write(
        tests / 'test_a.py',
        b'from cellacdc import widgets\nwidgets.myMessageBox()\n'
        b'getattr(widgets, "x" + "y")\n')
    _write(tests / 'test_b.py', b'import os\n')
    _write(tests / '.test_c.py', b'from cellacdc import apps\napps.Unknown\n')
    report_path = tmp_path / 'report.json'
    argv = ['--table', TABLE_PATH, '--no-src-root', '--report',
            str(report_path), str(tests)]
    assert rw.main(['--check'] + argv) == 1
    assert b'widgets.myMessageBox()' in changed.read_bytes()
    assert rw.main(argv) == 0
    assert b'widgets_message_box.myMessageBox()' in changed.read_bytes()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    files = {os.path.basename(key): value
             for key, value in report['files'].items()}
    assert sorted(files) == ['.test_c.py', 'test_a.py']
    assert files['test_a.py']['changed'] is True
    assert files['test_a.py']['references'][0]['new'] == (
        'widgets_message_box.myMessageBox')
    assert [u['kind'] for u in files['test_a.py']['unhandled']] == [
        'value_use']
    assert files['.test_c.py']['changed'] is False
    assert report['summary']['files_changed'] == 1
    assert rw.main(['--check'] + argv) == 0
    assert 'rewrite: 1 file(s) changed' in capsys.readouterr().out


def test_tool_parses_on_python_310():
    import ast
    with open(REWRITE_PATH, encoding='utf-8') as file:
        ast.parse(file.read(), feature_version=(3, 10))
    with open(__file__, encoding='utf-8') as file:
        ast.parse(file.read(), feature_version=(3, 10))


# -- C12 additions: reexports, unused imports, notebooks, patch lookups -----

def _table_with(aliases=None, **extra):
    with open(TABLE_PATH, encoding='utf-8') as file:
        data = json.load(file)
    data.update(extra)
    data['aliases'].update(aliases or {})
    return rw.NameTable(data)


def test_reexports_resolve_attributes_without_guessing():
    table = _table_with(reexports={
        'cellacdc.measurements_viewer': {
            'widgets': ['cellacdc.widgets', None]},
    }, aliases={'cellacdc.measurements_viewer.widgets': 'viewer_widgets'})
    # plan.py lists names that clash with a new submodule only as reexports
    del table.names['cellacdc.measurements_viewer']['widgets']
    new, report = run('''
        from cellacdc import measurements_viewer


        def test_x():
            return measurements_viewer.widgets.myMessageBox
        ''', table)
    assert 'widgets_message_box.myMessageBox' in new
    assert 'reexport_guessed' not in kinds(report)
    # a literal dotted path still means the new submodule
    ref, exact = table.resolve_attr(
        'cellacdc.measurements_viewer', 'widgets', literal=True)
    assert exact and ref.module == 'cellacdc.measurements_viewer.widgets'
    # with a reexports table, names the table omits are never guessed
    ref, _ = table.resolve_attr('cellacdc.gui', 'workers', literal=False)
    assert not getattr(ref, 'guessed', False)


def test_unused_imports_of_old_modules_are_dropped(table):
    source = '''
        from cellacdc import widgets
        from cellacdc import apps  # noqa: F401
        from cellacdc.apps import QDialogCombobox, QLineEditDialog
        from cellacdc import gui_navigation


        def f():
            return QLineEditDialog
        '''
    new, report = run(source, table, module='cellacdc.thing',
                      drop_unused=True, external_uses={})
    assert new == expect('''
        from cellacdc import apps  # noqa: F401
        from cellacdc.shared.dialogs.generic import QLineEditDialog


        def f():
            return QLineEditDialog
        ''')
    dropped = {(d['import'], d['local']) for d in report['dropped']}
    assert dropped == {
        ('cellacdc.widgets', 'widgets'),
        ('cellacdc.apps.QDialogCombobox', 'QDialogCombobox'),
        ('cellacdc.gui_navigation', 'gui_navigation'),
    }
    # another file reads cellacdc.thing.widgets: the import stays
    new, report = run(source, table, module='cellacdc.thing',
                      drop_unused=True,
                      external_uses={('cellacdc.thing', 'widgets'): {'x'}})
    assert 'from cellacdc import widgets' in new
    # without the use index nothing is dropped at module level
    new, _ = run(source, table, module='cellacdc.thing', drop_unused=True)
    assert 'from cellacdc import widgets' in new


def test_unused_imports_in_tests_keep_whole_moved_modules(table):
    new, report = run('''
        from cellacdc import gui_navigation
        from cellacdc import widgets

        __all__ = ['widgets']
        ''', table, drop_unused=True, external_uses={})
    assert new == expect('''
        from cellacdc.gui.mixins import navigation as gui_navigation
        from cellacdc import widgets

        __all__ = ['widgets']
        ''')
    assert report['dropped'] == []


def test_notebook_code_cells_are_rewritten(tmp_path, table):
    notebook = {
        'cells': [
            {'cell_type': 'markdown', 'metadata': {},
             'source': ['from cellacdc import widgets']},
            {'cell_type': 'code', 'execution_count': None, 'metadata': {},
             'outputs': [], 'source': [
                 '%matplotlib inline\n', 'from cellacdc import myutils']},
            {'cell_type': 'code', 'execution_count': None, 'metadata': {},
             'outputs': [], 'source': [
                 '!pip list\n', 'myutils.browse_url("x")\n', 'x = 1']},
        ],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
    }
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    path = _write(tmp_path / 'nb.ipynb',
                  text.replace('\n', '\r\n').encode('utf-8'))
    changed, report = rw.rewrite_file(str(path), table)
    assert changed, report
    data = path.read_bytes()
    assert data.count(b'\n') == data.count(b'\r\n')
    cells = json.loads(data.decode('utf-8'))['cells']
    assert cells[0]['source'] == ['from cellacdc import widgets']
    assert cells[1]['source'] == [
        '%matplotlib inline\n', 'from cellacdc.myutils import filesystem']
    assert cells[2]['source'] == [
        '!pip list\n', 'filesystem.browse_url("x")\n', 'x = 1']
    assert not rw.rewrite_file(str(path), table)[0]


def test_patch_in_a_loop_over_constant_names(indexed_table):
    new, report = run('''
        from cellacdc import myutils


        def test_x(monkeypatch, mocks):
            for name in ('check_install_package', 'subprocess'):
                monkeypatch.setattr(myutils, name, mocks[name])
        ''', indexed_table)
    assert 'monkeypatch.setattr(installs, name, mocks[name])' in new
    assert report['patches'][0]['form'] == 'loop'
    assert 'dynamic_patch' not in kinds(report)


def test_alias_assignment_of_an_old_split_module(table):
    new, report = run('''
        from cellacdc import apps


        def test_x():
            mod = apps
            mod.QDialogCombobox()
            return mod.SetMeasurementsDialog
        ''', table)
    assert new == expect('''
        from cellacdc.shared.dialogs import (
            generic as dialogs_generic,
            set_measurements as dialogs_set_measurements,
        )


        def test_x():
            dialogs_generic.QDialogCombobox()
            return dialogs_set_measurements.SetMeasurementsDialog
        ''')
    assert 'alias_assignment_removed' in kinds(report, 'info')


def test_assignment_patch_repeated_per_lookup_module(indexed_table):
    new, report = run('''
        from cellacdc import gui


        def test_x(menu):
            original = object()
            gui.QMenu = menu
            gui.QMenu = original
        ''', indexed_table)
    assert ('    # moved: QMenu is read in each of these modules\n'
            '    gui_toolbars.QMenu = menu\n' in new)
    assert '    gui_window.QMenu = menu\n' in new
    assert '    gui_window.QMenu = original\n' in new
    assert 'patch_multiple_lookups' not in kinds(report)


def test_reexported_name_patch_skips_the_defining_module(tmp_path):
    src = tmp_path / 'src'
    for rel, text in {
        'cellacdc/__init__.py': '',
        'cellacdc/viewer/__init__.py': '',
        'cellacdc/viewer/utils.py': 'def get_paths():\n    return 1\n',
        'cellacdc/viewer/config.py': (
            'from .utils import get_paths\n\n\n'
            'def load():\n    return get_paths()\n'),
        'cellacdc/viewer/window.py': '',
    }.items():
        _write(src / rel, text.encode('utf-8'))
    data = {
        'modules': {'cellacdc.old_viewer': 'cellacdc.viewer.window'},
        'names': {'cellacdc.old_viewer': {
            'get_paths': 'cellacdc.viewer.utils'}},
        'aliases': {'cellacdc.viewer.utils': 'viewer_utils',
                    'cellacdc.viewer.config': 'viewer_config',
                    'cellacdc.viewer.window': 'viewer_window'},
        'fanout': {'cellacdc.old_viewer': [
            'cellacdc.viewer.config', 'cellacdc.viewer.utils',
            'cellacdc.viewer.window']},
        'defined': {'cellacdc.old_viewer': []},
        'reexports': {},
    }
    table = rw.NameTable(data, index=rw.SourceIndex(str(src)))
    new, report = run('''
        from unittest.mock import Mock
        from cellacdc import old_viewer


        def test_x(monkeypatch):
            monkeypatch.setattr(old_viewer, 'get_paths', Mock())
        ''', table)
    assert "monkeypatch.setattr(viewer_config, 'get_paths', Mock())" in new
    assert 'viewer_utils' not in new


def test_imports_of_unchanged_modules_follow_their_old_bindings(tmp_path):
    _write(tmp_path / 'cellacdc' / '__init__.py', b'')
    _write(tmp_path / 'cellacdc' / 'service' / '__init__.py', b'')
    execution = _write(
        tmp_path / 'cellacdc' / 'service' / 'execution.py',
        b'from cellacdc import myutils\n\n\n'
        b'def run():\n    return myutils.check_install_package\n')
    _write(tmp_path / 'tests' / '__init__.py', b'')
    test = _write(
        tmp_path / 'tests' / 'test_x.py',
        b'from cellacdc.service import execution\n\n\n'
        b'def test_x(monkeypatch):\n'
        b'    monkeypatch.setattr(execution.myutils, "browse_url", None)\n'
        b'    return execution.run\n')
    table = rw.load_table(TABLE_PATH)
    files = [str(execution), str(test)]
    reexports = rw.find_reexports(files, table, rw.build_use_index(files))
    assert reexports == {'cellacdc.service.execution': {
        'myutils': 'cellacdc.myutils'}}
    table.add_reexports(reexports)
    assert rw.rewrite_file(str(test), table)[0]
    assert test.read_bytes() == (
        b'from cellacdc.service import execution\n'
        b'from cellacdc.myutils import filesystem\n\n\n'
        b'def test_x(monkeypatch):\n'
        b'    monkeypatch.setattr(filesystem, "browse_url", None)\n'
        b'    return execution.run\n')


def test_blame_ignore_revs_text():
    import importlib.util as util
    path = os.path.join(os.path.dirname(HERE), 'run.py')
    spec = util.spec_from_file_location('acdc_refactor_run_under_test', path)
    run_py = util.module_from_spec(spec)
    spec.loader.exec_module(run_py)
    shas = {'C12': 'a' * 40, 'C14': 'b' * 40}
    text = run_py.blame_ignore_text(shas)
    assert text.startswith(run_py.BLAME_BEGIN)
    assert ('a' * 40) in text and ('b' * 40) in text
    assert '\n# C12: ' in text and '\n# C14: ' in text
    kept = '# other\n' + 'c' * 40 + '\n'
    again = run_py.blame_ignore_text(
        {'C12': 'd' * 40, 'C14': 'e' * 40}, kept + '\n' + text)
    assert again.startswith(kept)
    assert ('a' * 40) not in again and ('d' * 40) in again
    assert again.count(run_py.BLAME_BEGIN) == 1


def test_code_strings_of_tests_are_rewritten(tmp_path, table):
    _write(tmp_path / 'tests' / '__init__.py', b'')
    source = (
        'import textwrap\n\n'
        'SCRIPT = textwrap.dedent("""\n'
        '    import cellacdc\n'
        '    from cellacdc import widgets\n\n'
        '    widgets.myMessageBox()\n'
        '""")\n'
        'LAYOUT = """\n'
        'import cellacdc.myutils as m\n'
        'if hasattr(m, "__path__"):\n'
        '    pass\n'
        'else:\n'
        '    from cellacdc import widgets\n'
        '"""\n'
    )
    path = _write(tmp_path / 'tests' / 'test_x.py',
                  source.replace('\n', '\r\n').encode('utf-8'))
    changed, report = rw.rewrite_file(str(path), table)
    assert changed
    new = path.read_bytes().decode('utf-8')
    assert new.count('\r\n') == new.count('\n')
    assert (
        '    import cellacdc\r\n'
        '    from cellacdc.shared.widgets import message_box as '
        'widgets_message_box\r\n\r\n'
        '    widgets_message_box.myMessageBox()\r\n') in new
    # the layout check of a model environment keeps its old branch
    assert '    from cellacdc import widgets\r\n"""' in new
    assert 'code_string_rewritten' in kinds(report, 'info')
    assert not rw.rewrite_file(str(path), table)[0]


def test_unused_plain_import_of_a_moved_module_uses_the_new_path(table):
    new, report = run('''
        import cellacdc.gui_navigation
        import cellacdc.widgets
        ''', table, drop_unused=True, external_uses={})
    assert new == expect('''
        import cellacdc.gui.mixins.navigation
        ''')
    assert [d['import'] for d in report['dropped']] == ['cellacdc.widgets']
