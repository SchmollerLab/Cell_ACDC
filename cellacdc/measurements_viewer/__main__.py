"""Runs the measurements viewer with python -m on the measurements_viewer
package, optionally loading the folder given as the first argument (run
of main_window/window.py).
"""
import sys

from .main_window import window as viewer_window

if __name__ == '__main__':
    viewer_window.run(sys.argv[1] if len(sys.argv) > 1 else None)
