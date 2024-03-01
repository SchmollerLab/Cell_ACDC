UPGRADE_BTRACK = False

try:
    import btrack
    from cellacdc.myutils import get_package_version 
    version = get_package_version('btrack')  
    minor = version.split('.')[1]
    if int(minor) < 5:
        UPGRADE_BTRACK = True
except Exception as e:
    pass

from cellacdc import myutils

myutils.check_install_package(
    'Bayesian Tracker',
    import_pkg_name='btrack',
    pypi_name='btrack', 
    force_upgrade=UPGRADE_BTRACK
)
