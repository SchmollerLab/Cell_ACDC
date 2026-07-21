def voxel_display_scale(dz: float, dy: float, dx: float) -> tuple[float, float, float]:
    """Return vispy ``STTransform`` scale for ``(Z, Y, X)`` data (Cell-ACDC style)."""
    dx_eff = dx if dx > 0 else 1.0
    dy_eff = dy if dy > 0 else 1.0
    dz_eff = dz if dz > 0 else 1.0
    return (1.0, dy_eff / dx_eff, dz_eff / dx_eff)