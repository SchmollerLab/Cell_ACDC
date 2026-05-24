"""View-model commands for UI-neutral formatting helpers."""

from __future__ import annotations

from cellacdc.domain.display_images import distant_gray, rgb_to_gray
from cellacdc.myutils import (
    _bytes_to_GB,
    get_chname_from_basename,
    get_number_fstring_formatter,
    get_salute_string,
    seconds_to_ETA,
)


class FormattingViewModel:
    """Application-facing commands for display string formatting."""

    def number_fstring_formatter(self, dtype, *, precision=4):
        return get_number_fstring_formatter(dtype, precision=precision)

    def channel_name_from_basename(
        self,
        filename,
        basename,
        *,
        remove_ext=True,
    ):
        return get_chname_from_basename(
            filename,
            basename,
            remove_ext=remove_ext,
        )

    def bytes_to_gb(self, size_bytes):
        return _bytes_to_GB(size_bytes)

    def seconds_to_eta(self, seconds):
        return seconds_to_ETA(seconds)

    def salute_string(self):
        return get_salute_string()

    def distant_gray(
        self,
        desired_gray,
        background_gray,
        *,
        threshold=0.3,
    ):
        return distant_gray(
            desired_gray,
            background_gray,
            threshold=threshold,
        )

    def rgb_to_gray(self, red, green, blue):
        return rgb_to_gray(red, green, blue)
