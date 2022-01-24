import os
import errno

from shutil import copytree
from os.path import isdir, join
from fnmatch import fnmatch, filter

from detectron2.config import CfgNode as CN

def include_patterns(*patterns):
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns


def remove_empty_dirs(output_folder):
    dirs = [x[0] for x in os.walk(output_folder, topdown=False)]
    for dir in dirs:
        try:
            os.rmdir(dir)
        except Exception as e:
            if e.errno == errno.ENOTEMPTY:
                print("Directory: {0} not empty".format(dir))


def copy_code(path):
    path = os.path.join(path, 'cellacdc')
    py_files_path = os.path.dirname(os.path.realpath(__file__))
    copytree(py_files_path, path, ignore=include_patterns('*.py', '*.yaml'))
    remove_empty_dirs(path)

def initialize_new_config_values(cfg):
    cfg.INPUT_DIR = None

    cfg.MODEL.ROI_HEADS.NUM_MASK_CLASSES = None

    cfg.INPUT.CROP_SIZE = None

    cfg.POSTPROCESSING = CN()
    cfg.POSTPROCESSING.POSSIBLE_COMPS = None
    cfg.POSTPROCESSING.OPTIONAL_OBJECT_SCORE_THRESHOLD = None
    cfg.POSTPROCESSING.PARENT_OVERRIDE_THRESHOLD = None

    return cfg