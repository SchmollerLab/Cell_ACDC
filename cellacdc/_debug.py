import inspect

from . import printl

def print_all_callers(self):
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe, 2)
    outerframes_format = '\n'
    for frame in outerframes:
        outerframes_format = f'{outerframes_format}  * {frame.function}\n'
    printl(outerframes_format)