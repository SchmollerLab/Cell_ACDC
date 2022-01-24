from setuptools import setup

setup(
    name="cellacdc",
    version="1.2.3",
    license="BSD",
    author="Francesco Padovani and Benedikt Mairhoermann",
    author_email="francesco.padovani@helmholtz-muenchen.de",
    description="segmentation, tracking and image annotations",
    entry_points = {
            'console_scripts': ['acdc = cellacdc.main:main']
        },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent"
    )
)
