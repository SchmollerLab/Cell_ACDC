from cellacdc import myutils

myutils.check_install_package(
    'cellSAM',
    pypi_name='git+https://github.com/vanvalenlab/cellSAM.git',
    import_pkg_name='cellSAM'
)

import cellSAM

# Available CellSAM model types
# cellsam_general: trained on datasets from the original publication
# cellsam_extra: incorporates additional datasets beyond the paper
model_types = {
    'General': 'cellsam_general',
    'Extra': 'cellsam_extra',
}
