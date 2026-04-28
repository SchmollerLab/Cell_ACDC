# only needed for cython extensions, not needed to run normally
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        Extension(
            "cellacdc.precompiled.precompiled_functions",
            sources=["cellacdc/precompiled_functions.pyx"],
            include_dirs=[np.get_include()],
        ),
        annotate=True,
        build_dir="build/cython",  # .c and .html files go here
    )
)
# # move compiled binary to precompiled/
# import shutil
# import os

# src_dir = "cellacdc"
# for filename in os.listdir(src_dir):
#     if filename.startswith("precompiled_functions") and (filename.endswith(".so") or filename.endswith(".pyd")):
#         target_path = os.path.join("cellacdc", "precompiled", filename)
#         shutil.move(os.path.join(src_dir, filename), target_path)
#         print(f"Moved {filename} to {target_path}")

