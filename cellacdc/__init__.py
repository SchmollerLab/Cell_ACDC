import sys
import os
import inspect

def printl(*objects, **kwargs):
    cf = inspect.currentframe()
    filpath = inspect.getframeinfo(cf).filename
    filename = os.path.basename(filpath)
    print('*'*30)
    print(f'File "{filename}", line {cf.f_back.f_lineno}:')
    print(*objects, **kwargs)
    print('='*30)

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(cellacdc_path, 'temp')

if not os.path.exists(temp_path):
    os.makedirs(temp_path)

try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception as e:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "not-installed"

__author__ = 'Francesco Padovani and Benedikt Mairhoermann'

cite_url = 'https://www.biorxiv.org/content/10.1101/2021.09.28.462199v2'

# Initialize variables that need to be globally accessible
base_cca_df = {
    'cell_cycle_stage': 'G1',
    'generation_num': 2,
    'relative_ID': -1,
    'relationship': 'mother',
    'emerg_frame_i': -1,
    'division_frame_i': -1,
    'is_history_known': False,
    'corrected_assignment': False
}

base_acdc_df = {
    'is_cell_dead': False,
    'is_cell_excluded': False,
    'was_manually_edited': 0
}

is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))

yeaz_weights_filenames = [
    'unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5',
    'weights_budding_BF_multilab_0_1.hdf5'
]

yeastmate_weights_filenames = [
    'yeastmate_advanced.yaml',
    'yeastmate_weights.pth',
    'yeastmate.yaml'
]
