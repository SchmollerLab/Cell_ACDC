"""Prepare preselected segmentation models in isolated environments."""

import argparse
from pathlib import Path

from .environment import get_or_create_model_env
from .prompts import AutoHandler, CliHandler


MODEL_ALIASES = {
    'cellpose': 'cellpose_v4',
    'cellpose3': 'cellpose_v3',
    'stardist': 'StarDist',
    'yeaz': 'YeaZ_v2',
    'deepsea': 'DeepSea',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    models_path = Path(__file__).resolve().parents[1] / 'models'
    models = {
        path.parent.name for path in models_path.glob('*/acdcSegment.py')
        if not path.parent.name.startswith('_')
    }
    parser.add_argument(
        'models', nargs='+', choices=sorted(models | MODEL_ALIASES.keys()),
        help='Model folder names or pip extra names.',
    )
    parser.add_argument(
        '--yes', action='store_true',
        help='Use automatically detected defaults instead of prompting.',
    )
    args = parser.parse_args(argv)
    handler = AutoHandler() if args.yes else CliHandler()
    selected = dict.fromkeys(MODEL_ALIASES.get(name, name) for name in args.models)
    for model in selected:
        print(f'Preparing isolated environment for {model}...', flush=True)
        python = get_or_create_model_env(
            model, auto_install=True, prompt_func=handler.ask,
        )
        print(f'{model}: {python}', flush=True)


if __name__ == '__main__':
    main()
