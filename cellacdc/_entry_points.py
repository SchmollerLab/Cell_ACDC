"""Entry points that parse their own command line (standard library only).

``cellacdc.config`` parses ``sys.argv`` with the ``acdc`` parser when it is
imported. For the programs listed here that parse is skipped entirely, so
their own ``-h``, ``-i``, ``-p`` ... are not taken (or rejected) by the
``acdc`` parser.
"""
from __future__ import annotations

import sys
from typing import Iterator, Sequence

OWN_PARSER_SCRIPTS = {
    'acdc-setup-models': 'cellacdc.service.setup_models',
    'acdc-data': 'cellacdc.data_cli',
}
"""Console script name -> module run by ``python -m``."""
OWN_PARSER_MODULES = (
    frozenset(OWN_PARSER_SCRIPTS.values())
    | {'cellacdc.acdc_datastruct._worker'}
)
_LAUNCHER_SUFFIXES = ('-script.pyw', '-script.py', '.exe')


def script_name(argv0: str) -> str:
    """Lower-cased program name of ``argv[0]``: basename (both '/' and '\\'
    separators) without a trailing '.exe', '-script.py' or '-script.pyw'
    (case-insensitive), e.g. 'C:\\x\\Scripts\\ACDC-DATA.EXE' gives
    'acdc-data'."""
    name = argv0.replace('\\', '/').rsplit('/', 1)[-1].lower()
    for suffix in _LAUNCHER_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _dash_m_modules(orig_argv: Sequence[str]) -> Iterator[str]:
    """Module names given as '-m X' or '-mX' anywhere in ``orig_argv``."""
    for i, arg in enumerate(orig_argv):
        if arg == '-m' and i + 1 < len(orig_argv):
            yield orig_argv[i + 1]
        elif arg.startswith('-m') and len(arg) > 2:
            yield arg[2:]


def parses_own_args(
        argv: Sequence[str] | None = None,
        orig_argv: Sequence[str] | None = None
    ) -> bool:
    """True when the running program parses its own command line.

    That is the case when ``script_name(argv[0])`` is a key of
    ``OWN_PARSER_SCRIPTS`` (exact match, so 'acdc-database' does not
    count), or when ``orig_argv`` runs one of ``OWN_PARSER_MODULES`` with
    ``-m``. ``argv`` defaults to ``sys.argv`` and ``orig_argv`` to
    ``sys.orig_argv``; empty or missing values are tolerated.
    (Known harmless false positive: ``python -c "..." -m cellacdc.data_cli``.)
    """
    if argv is None:
        argv = getattr(sys, 'argv', None) or []
    if orig_argv is None:
        orig_argv = getattr(sys, 'orig_argv', None) or []
    if argv and script_name(str(argv[0])) in OWN_PARSER_SCRIPTS:
        return True
    return any(
        module in OWN_PARSER_MODULES
        for module in _dash_m_modules([str(arg) for arg in orig_argv])
    )
