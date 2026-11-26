"""Exception hierarchy of the data-structure core.

``UsageError`` (and its subclasses) is raised only before any file-system
change: the command line maps it to exit code 2 and guarantees that the
source and destination folders were not changed. Every other
``DataStructError`` is a runtime failure (exit code 1), except
``ConversionCancelled`` (exit code 130).
"""
from __future__ import annotations

from typing import Iterable, Sequence


class DataStructError(Exception):
    """Base class of every error raised by the data-structure core.

    Parameters
    ----------
    message : str
        One-line (or short multi-line) description for the user.
    hint : str, optional
        What the user can do about it (e.g. the command line flag to use).
    details : str, optional
        Longer text such as a traceback or a list of files.
    """
    def __init__(
            self, message: str, *, hint: str | None = None,
            details: str | None = None
        ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details


class UsageError(DataStructError, ValueError):
    """The request was rejected before any file-system change."""


class SourceError(UsageError):
    """Missing, ambiguous, unsupported or invalid source files."""


class FilenamePatternError(SourceError):
    """One-file-per-channel file names do not follow
    ``<basename><position>_<channel>``.

    ``files`` lists the offending file names.
    """
    def __init__(
            self, message: str, *, files: Iterable[str] = (),
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.files = list(files)


class MissingChannelFilesError(SourceError):
    """One-file-per-channel: files of saved (position, channel) pairs are
    missing. ``missing`` lists ``(position id, raw channel token)``."""
    def __init__(
            self, message: str, *,
            missing: Iterable[tuple[int, str]] = (),
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.missing = list(missing)


class SettingsError(UsageError):
    """Invalid conversion settings. ``problems`` lists every problem found
    (defaults to ``[message]``)."""
    def __init__(
            self, message: str, *, problems: Iterable[str] | None = None,
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.problems = [message] if problems is None else list(problems)


class PlanError(UsageError):
    """The plan cannot be executed (move/rename targets exist, unknown
    positions, ...)."""


class ExistingPositionsError(PlanError, FileExistsError):
    """Position folders that would be written already exist.

    ``existing`` lists their numbers, ``choices`` the accepted
    ``--on-existing`` values.
    """
    def __init__(
            self, message: str, *, existing: Iterable[int] = (),
            choices: Sequence[str] = ('overwrite', 'add', 'create-new'),
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.existing = list(existing)
        self.choices = tuple(choices)


class RestructureOptionsError(UsageError):
    """Invalid "Re-structure image files" options or recipe."""


class MissingDependencyError(DataStructError):
    """Reader packages are missing and could not be installed."""
    def __init__(
            self, message: str, *, packages: Iterable[str] = (),
            install_command: str = '', hint: str | None = None,
            details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.packages = list(packages)
        self.install_command = install_command


class MetadataReadError(DataStructError):
    """Reading the metadata of ``filename`` failed (``details`` holds the
    worker traceback)."""
    def __init__(
            self, message: str, *, filename: str = '',
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.filename = filename


class MoveRawFilesError(DataStructError):
    """Renaming or moving the raw files failed.

    ``moved`` lists the ``(src, dst)`` moves already done (they stay done),
    ``failed`` the path that could not be moved and ``permission`` whether
    the cause was a ``PermissionError``.
    """
    def __init__(
            self, message: str, *, moved: Iterable[tuple[str, str]] = (),
            failed: str | None = None, permission: bool = False,
            hint: str | None = None, details: str | None = None
        ) -> None:
        super().__init__(message, hint=hint, details=details)
        self.moved = list(moved)
        self.failed = failed
        self.permission = permission


class ConversionCancelled(DataStructError):
    """The user cancelled the run."""
