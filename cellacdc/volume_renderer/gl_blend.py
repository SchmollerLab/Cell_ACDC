"""OpenGL blend state for vispy volume layers (napari-compatible).

Napari composites multiple MIP fluor channels by giving the first visible
channel ``translucent_no_depth`` and additional channels ``additive``, with a
special case for the bottommost visible layer so additive does not blend
against stale framebuffer alpha (see napari ``_vispy/layers/base.py``).
"""

from __future__ import annotations

from typing import Literal

VolumeBlending = Literal[
    "translucent",
    "translucent_no_depth",
    "additive",
]

# Mirrors napari._vispy.utils.gl.BLENDING_MODES (subset used for volumes).
_BLENDING_MODES: dict[VolumeBlending, dict] = {
    "translucent": {
        "depth_test": True,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "one_minus_src_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
    "translucent_no_depth": {
        "depth_test": False,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "one_minus_src_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
    "additive": {
        "depth_test": False,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "dst_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
}


def volume_gl_state(
    blending: VolumeBlending,
    *,
    first_visible: bool,
) -> dict:
    """Return kwargs for ``vispy`` ``set_gl_state`` for a volume visual."""
    state = dict(_BLENDING_MODES[blending])
    if not first_visible:
        return state

    # Bottommost visible layer: avoid pathological blending with the canvas.
    if blending == "additive":
        src_color, dst_color = "src_alpha", "zero"
    else:
        src_color, dst_color = "src_alpha", "one_minus_src_alpha"
    return {
        "depth_test": state["depth_test"],
        "cull_face": False,
        "blend": True,
        "blend_func": (src_color, dst_color, "one", "one"),
        "blend_equation": "func_add",
    }
