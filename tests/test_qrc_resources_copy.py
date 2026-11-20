"""Tests for the copy of qrc_resources.py done when cellacdc is imported.

Several processes can import cellacdc at the same time. The file must only
appear complete, otherwise a process can import a half-written file. All
tests work in `tmp_path`: they never touch the qrc_resources.py of the
package or of the user profile.
"""
import os
import shutil
import stat
import sys
import time

import pytest

from cellacdc import _copy_qrc_resources_file, _write_text_file_atomic

# CRLF line endings and non-ASCII bytes check that the copy is exact
SRC_CONTENT = (
    b'from qtpy import QtCore\r\n\r\nqt_resource_data = b"\\x00\\xff"\r\n'
    + bytes(range(256)) * 64
)


@pytest.fixture
def paths(tmp_path):
    src_path = tmp_path / 'qrc_resources_light.py'
    src_path.write_bytes(SRC_CONTENT)
    install_folder = tmp_path / 'install'
    install_folder.mkdir()
    user_folder = tmp_path / 'user'
    user_folder.mkdir()
    return (
        str(src_path),
        str(install_folder / 'qrc_resources.py'),
        str(user_folder / 'qrc_resources.py'),
    )


def _read_bytes(filepath):
    with open(filepath, 'rb') as file:
        return file.read()


def test_destination_appears_only_complete(paths, monkeypatch):
    src_path, dst_path, user_dst_path = paths
    real_replace = os.replace
    calls = []

    def checking_replace(tmp_path, replaced_path):
        # Until the replace, readers of dst_path find no file at all
        assert not os.path.exists(dst_path)
        assert os.path.dirname(tmp_path) == os.path.dirname(dst_path)
        assert _read_bytes(tmp_path) == SRC_CONTENT
        calls.append((tmp_path, replaced_path))
        real_replace(tmp_path, replaced_path)

    monkeypatch.setattr(os, 'replace', checking_replace)

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert has_admin_rights
    assert len(calls) == 1
    assert calls[0][1] == dst_path
    assert _read_bytes(dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(dst_path)) == ['qrc_resources.py']
    assert not os.path.exists(user_dst_path)


def test_failed_copy_leaves_no_partial_file(paths, monkeypatch):
    src_path, dst_path, user_dst_path = paths
    install_folder = os.path.dirname(dst_path)
    real_copyfile = shutil.copyfile

    def copyfile_failing_halfway(src, dst, *args, **kwargs):
        if os.path.dirname(dst) != install_folder:
            return real_copyfile(src, dst, *args, **kwargs)
        with open(dst, 'wb') as file:
            file.write(SRC_CONTENT[:100])
        raise OSError('No space left on device')

    monkeypatch.setattr(shutil, 'copyfile', copyfile_failing_halfway)

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert not has_admin_rights
    assert os.listdir(install_folder) == []
    assert _read_bytes(user_dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(user_dst_path)) == ['qrc_resources.py']


def _replace_denied(tmp_path, replaced_path):
    # What Windows does when another process has the destination open, or
    # when the destination is read-only
    raise PermissionError(13, 'Access is denied', replaced_path)


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(time, 'sleep', calls.append)
    return calls


def test_destination_written_by_other_process_is_kept(
    paths, monkeypatch, sleeps
):
    src_path, dst_path, user_dst_path = paths
    with open(dst_path, 'wb') as file:
        file.write(SRC_CONTENT)
    monkeypatch.setattr(os, 'replace', _replace_denied)

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert has_admin_rights
    assert sleeps == []
    assert _read_bytes(dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(dst_path)) == ['qrc_resources.py']
    assert not os.path.exists(user_dst_path)


def test_destination_that_cannot_be_replaced_falls_back(
    paths, monkeypatch, sleeps
):
    src_path, dst_path, user_dst_path = paths
    outdated_content = SRC_CONTENT[::-1]
    with open(dst_path, 'wb') as file:
        file.write(outdated_content)
    real_replace = os.replace

    def replace_denied_in_install_folder(tmp_path, replaced_path):
        if replaced_path == dst_path:
            _replace_denied(tmp_path, replaced_path)
        real_replace(tmp_path, replaced_path)

    monkeypatch.setattr(os, 'replace', replace_denied_in_install_folder)

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert not has_admin_rights
    assert len(sleeps) > 0
    assert _read_bytes(dst_path) == outdated_content
    assert os.listdir(os.path.dirname(dst_path)) == ['qrc_resources.py']
    assert _read_bytes(user_dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(user_dst_path)) == ['qrc_resources.py']


@pytest.mark.skipif(
    sys.platform != 'win32',
    reason='Only Windows refuses to replace a read-only file'
)
def test_read_only_destination_falls_back(paths, sleeps):
    src_path, dst_path, user_dst_path = paths
    outdated_content = SRC_CONTENT[::-1]
    with open(dst_path, 'wb') as file:
        file.write(outdated_content)
    os.chmod(dst_path, stat.S_IREAD)
    try:
        has_admin_rights = _copy_qrc_resources_file(
            src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
        )
        assert _read_bytes(dst_path) == outdated_content
    finally:
        os.chmod(dst_path, stat.S_IREAD | stat.S_IWRITE)

    assert not has_admin_rights
    assert os.listdir(os.path.dirname(dst_path)) == ['qrc_resources.py']
    assert _read_bytes(user_dst_path) == SRC_CONTENT


def test_permission_error_without_destination_falls_back(
    paths, monkeypatch, sleeps
):
    src_path, dst_path, user_dst_path = paths
    real_replace = os.replace

    def replace_denied_in_install_folder(tmp_path, replaced_path):
        if replaced_path == dst_path:
            _replace_denied(tmp_path, replaced_path)
        real_replace(tmp_path, replaced_path)

    monkeypatch.setattr(os, 'replace', replace_denied_in_install_folder)

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert not has_admin_rights
    assert os.listdir(os.path.dirname(dst_path)) == []
    assert _read_bytes(user_dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(user_dst_path)) == ['qrc_resources.py']


def test_missing_install_folder_falls_back_to_user_folder(tmp_path, paths):
    src_path, _, user_dst_path = paths
    dst_path = str(tmp_path / 'missing' / 'qrc_resources.py')

    has_admin_rights = _copy_qrc_resources_file(
        src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
    )

    assert not has_admin_rights
    assert not os.path.exists(os.path.dirname(dst_path))
    assert _read_bytes(user_dst_path) == SRC_CONTENT
    assert os.listdir(os.path.dirname(user_dst_path)) == ['qrc_resources.py']


def test_missing_source_raises_and_leaves_no_files(tmp_path, paths):
    _, dst_path, user_dst_path = paths
    src_path = str(tmp_path / 'missing_qrc_resources_light.py')

    with pytest.raises(FileNotFoundError):
        _copy_qrc_resources_file(
            src_path, dst_path, user_dst_qrc_resources_path=user_dst_path
        )

    assert os.listdir(os.path.dirname(dst_path)) == []
    assert os.listdir(os.path.dirname(user_dst_path)) == []


def test_text_rewrite_replaces_file_in_one_step(tmp_path, monkeypatch):
    dst_path = str(tmp_path / 'qrc_resources.py')
    old_text = 'from PyQt5 import QtCore\n\nqt_resource_data = b""\n'
    new_text = old_text.replace('from PyQt5', 'from qtpy')
    with open(dst_path, 'w') as file:
        file.write(old_text)
    real_replace = os.replace
    calls = []

    def checking_replace(tmp_file_path, replaced_path):
        # Until the replace, readers of dst_path find the old file intact
        with open(dst_path, 'r') as file:
            assert file.read() == old_text
        with open(tmp_file_path, 'r') as file:
            assert file.read() == new_text
        calls.append(replaced_path)
        real_replace(tmp_file_path, replaced_path)

    monkeypatch.setattr(os, 'replace', checking_replace)

    _write_text_file_atomic(dst_path, new_text)

    assert calls == [dst_path]
    with open(dst_path, 'r') as file:
        assert file.read() == new_text
    assert os.listdir(tmp_path) == ['qrc_resources.py']


def _write_text(filepath, text):
    with open(filepath, 'w') as file:
        file.write(text)


def _read_text(filepath):
    with open(filepath, 'r') as file:
        return file.read()


def test_text_rewrite_waits_while_file_is_open(
    tmp_path, monkeypatch, sleeps
):
    dst_path = str(tmp_path / 'qrc_resources.py')
    old_text = 'from PyQt5 import QtCore\n'
    new_text = 'from qtpy import QtCore\n'
    _write_text(dst_path, old_text)
    real_replace = os.replace
    calls = []

    def replace_denied_twice(tmp_file_path, replaced_path):
        # Another process reads the file during the first two attempts
        calls.append(replaced_path)
        if len(calls) <= 2:
            _replace_denied(tmp_file_path, replaced_path)
        real_replace(tmp_file_path, replaced_path)

    monkeypatch.setattr(os, 'replace', replace_denied_twice)

    _write_text_file_atomic(dst_path, new_text)

    assert len(calls) == 3
    assert len(sleeps) == 2
    assert _read_text(dst_path) == new_text
    assert os.listdir(tmp_path) == ['qrc_resources.py']


def test_text_rewrite_raises_when_file_cannot_be_replaced(
    tmp_path, monkeypatch, sleeps
):
    dst_path = str(tmp_path / 'qrc_resources.py')
    old_text = 'from PyQt5 import QtCore\n'
    _write_text(dst_path, old_text)
    monkeypatch.setattr(os, 'replace', _replace_denied)

    with pytest.raises(PermissionError):
        _write_text_file_atomic(dst_path, 'from qtpy import QtCore\n')

    assert _read_text(dst_path) == old_text
    assert os.listdir(tmp_path) == ['qrc_resources.py']
