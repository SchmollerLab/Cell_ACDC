"""Fake myutils/filesystem.py: reads printl as a global."""
from .. import printl


def browse_url(url):
    printl(url)
