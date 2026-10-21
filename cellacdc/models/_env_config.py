"""Isolated environment configuration of the models.

When a model runs in its own environment (``_acdc_<model folder name>``),
this file defines how that environment is built. To add a model, add an
entry to ``MODEL_ENVS`` using the model folder name as key. Models without
an entry get the defaults (host Python version, no extra packages).

Entry fields (all optional):
    python: Python version of the environment, e.g. "3.11". Default is the
        Python version of the main Cell-ACDC environment.
    torch: True if the model needs PyTorch. It is installed automatically
        with the build matching the machine
    packages: list of packages to install. Each item is either a string
        (pip name, also used as import name) or a dict with the keys
        accepted by ``myutils.check_install_package``
        (``pkg_name``, ``import_pkg_name``, ``pypi_name``,
        ``min_version``, ``max_version``, ...). ``distribution_name`` is
        also accepted; it is only used to detect an existing installation.
    main_env_packages: optional package specs used only to detect an existing
        main-environment installation, without importing any model modules.
        Defaults to packages; torch/torchvision are also checked when torch is
        True. Use distribution_name for a distribution whose install source is
        a URL. Detection checks metadata and versions, not runtime importability.
    setup_steps: ordered steps executed after torch and packages. Each has a
        name, optional description, commands (lists of arguments appended to
        the model environment's Python, e.g. ["-m", "pip", "install", URL]),
        and optional check (Python code). A successful check skips the step;
        after setup the check must succeed. Output is streamed and errors raised.
        Commands never use a shell. Without a check, a step runs on every setup.
    required_env: environment variables the model needs at runtime (e.g.
        access tokens). Each item is a dict with ``name``, optional
        ``description``, ``secret`` (mask the input) and
        ``unless_init_kwargs`` (list of init kwargs; if any of them is set,
        the variable is not needed). Missing variables are asked for before
        the model starts and kept only in memory for the current session.

This file must stay free of imports from ``cellacdc`` because it is loaded
by path, without importing the model packages.
"""

# PyPI only has pomBseen 0.2.0b0, whose `pomBseg` lacks the newer arguments
# (e.g. `apply_convex_hull`) that the Cell-ACDC wrapper passes.
_POMBSEEN_GIT_STEP = {
    'name': 'pomBseen from GitHub',
    'description': 'Install the current pomBseen version from GitHub.',
    'check': 'import inspect, pombseen.main as m\n'
             'assert "apply_convex_hull" in '
             'inspect.signature(m.pomBseg).parameters',
    'commands': [
        ['-m', 'pip', 'install', '--force-reinstall', '--no-deps',
         'git+https://github.com/Teranis/pomBseen.git'],
    ],
}

MODEL_ENVS = {
    'thresholding': {},
    'skip_segmentation': {},
    'cellpose_v2': {
        'torch': True,
        'packages': [
            {'pkg_name': 'cellpose', 'min_version': '2.3.2',
             'max_version': '3.0',
             'include_lower_version': True},
        ],
    },
    'cellpose_v3': {
        'torch': True,
        'python': '3.11',
        'main_env_packages': [
            {'pkg_name': 'cellpose', 'min_version': '3.1.1.2',
             'max_version': '4.0', 'include_lower_version': True},
        ],
        'packages': [
            {'pkg_name': 'cellpose', 'min_version': '3.1.1.2',
             'max_version': '4.0',
             'include_lower_version': True,},
        ],
    },
    'cellpose_v4': {
        'torch': True,
        'packages': [
            {'pkg_name': 'cellpose', 'min_version': '4.0.6',
             'include_lower_version': True},
        ],
    },
    'Cellpose_germlineNuclei': {
        'torch': True,
        'packages': [
            {'pkg_name': 'cellpose', 'min_version': '2.3.2',
             'max_version': '3.0',
             'include_lower_version': True},
        ],
    },
    'StarDist': {
        'python': '3.11',
        'packages': [
            {'pkg_name': 'tensorflow', 'max_version': '2.19.2'},
            {'pkg_name': 'numpy', 'max_version': '2.0.0'},
            {'pkg_name': 'tifffile', 'max_version': '2025'},
            {'pkg_name': 'scipy', 'max_version': '1.14'},
            'stardist',
        ],
    },
    'YeaZ': {
        'python': '3.11',
        'packages': [{'pkg_name': 'tensorflow', 'max_version': '2.17'}],
    },
    'YeaZ_v2': {
        'torch': True,
        'packages': ['yeaz'],
    },
    'DeepSea': {
        'torch': True,
        'packages': ['deepsea', 'munkres'],
    },
    'YeastMate': {
        'torch': True,
        'main_env_packages': ['detectron2', 'yeastmatedetector'],
        'setup_steps': [
            {
                'name': 'detectron2',
                'description': 'Build detectron2 v0.5. Windows requires the '
                               'Visual C++ build tools; Git must be installed.',
                'check': 'import detectron2',
                'commands': [
                    ['-m', 'pip', 'install', 'Cython'],
                    ['-m', 'pip', 'install', '--no-build-isolation',
                     'git+https://github.com/facebookresearch/detectron2.git@v0.5'],
                ],
            },
            {
                'name': 'YeastMate and headless OpenCV',
                'check': 'import yeastmatedetector\n'
                         'from importlib.metadata import distributions\n'
                         'names = {d.metadata["Name"].lower() for d in distributions()}\n'
                         'assert "opencv-python" not in names\n'
                         'assert "opencv-python-headless" in names',
                'commands': [
                    ['-m', 'pip', 'install',
                     'git+https://github.com/hoerlteam/YeastMate.git'],
                    ['-m', 'pip', 'uninstall', '-y',
                     'opencv-python', 'opencv-python-headless'],
                    ['-m', 'pip', 'install', 'opencv-python-headless'],
                ],
            },
        ],
    },
    'InstanSeg': {
        'torch': True,
        'packages': [
            {'pkg_name': 'InstanSeg', 'import_pkg_name': 'instanseg',
             'pypi_name': 'instanseg-torch'},
        ],
    },
    'sam2': {
        'torch': True,
        'packages': ['sam2'],
    },
    'segment_anything': {
        'torch': True,
        'packages': ['segment_anything'],
    },
    'nnInteractive': {
        'python': '3.12',
        'torch': True,
        'packages': ['nnInteractive', 'huggingface_hub'],
    },
    'cellsam': {
        'torch': True,
        'main_env_packages': [{'pkg_name': 'cellSAM'}],
        'packages': [
            {'pkg_name': 'cellSAM', 'import_pkg_name': 'cellSAM',
             'pypi_name': 'git+https://github.com/vanvalenlab/cellSAM.git'},
        ],
        'required_env': [
            {
                'name': 'DEEPCELL_ACCESS_TOKEN',
                'description': (
                    'CellSAM downloads its pretrained models from DeepCell, '
                    'which requires a personal access token.\n'
                    'Create one (free) at https://users.deepcell.org and '
                    'paste it below.\n\n'
                    'The token is only kept for this session. To avoid this '
                    'prompt, set the DEEPCELL_ACCESS_TOKEN environment '
                    'variable.'
                ),
                'secret': True,
                'unless_init_kwargs': ['model_path'],
            },
        ],
    },
    'omnipose': {
        'torch': True,
        'packages': [
            {'pkg_name': 'omnipose', 'pypi_name': 'omnipose_acdc'},
        ],
    },
    'omnipose_custom': {
        'torch': True,
        'packages': [
            {'pkg_name': 'omnipose_acdc', 'import_pkg_name': 'omnipose'},
        ],
    },
    'BABY': {
        'python': '3.10',
        'packages': [
            # TF 2.13 pins typing-extensions<4.6, incompatible with the HTTP runtime.
            {'pkg_name': 'tensorflow', 'min_version': '2.12',
             'include_lower_version': True, 'max_version': '2.13'},
            {'pkg_name': 'numpy', 'max_version': '1.24'},
            {'pkg_name': 'scikit-image', 'import_pkg_name': 'skimage',
             'max_version': '0.25'},
            {'pkg_name': 'opencv-python-headless', 'import_pkg_name': 'cv2',
             'max_version': '4.12'},
            {'pkg_name': 'typing-extensions', 'import_pkg_name': 'typing_extensions',
             'min_version': '4.16', 'include_lower_version': True},
            {'pkg_name': 'baby', 'pypi_name': 'baby-seg'},
        ],
    },
    'delta': {
        'packages': [{'pkg_name': 'delta', 'pypi_name': 'delta2'}],
    },
    'pomBseen': {
        'packages': [{'pkg_name': 'pombseen', 'pypi_name': 'pomBseen'}],
        'setup_steps': [_POMBSEEN_GIT_STEP],
    },
    'pomBseen_nuclear': {
        'packages': [{'pkg_name': 'pombseen', 'pypi_name': 'pomBseen'}],
        'setup_steps': [_POMBSEEN_GIT_STEP],
    },
}
