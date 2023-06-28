from . import _core

def add_rotational_volume_regionprops(
        rp, PhysicalSizeY=1, PhysicalSizeX=1, logger_func=None
    ):
    for obj in rp:
        vol_vox, vol_fl = _core._calc_rotational_vol(
            obj, PhysicalSizeY, PhysicalSizeX, logger=logger_func
        )
        obj.vol_vox, obj.vol_fl = vol_vox, vol_fl
    return rp