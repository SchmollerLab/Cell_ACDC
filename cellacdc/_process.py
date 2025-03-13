import subprocess
import sys

from subprocess import Popen, PIPE, STDOUT

import multiprocessing
import argparse

ap = argparse.ArgumentParser(
    prog='Cell-ACDC process', description='Used to spawn a separate process', 
    formatter_class=argparse.RawTextHelpFormatter
)
ap.add_argument(
    '-c', '--command', required=True, type=str, metavar='COMMAND',
    help='String of commands separated by comma.'
)

ap.add_argument(
    '-l', '--log_filepath',
    default='',
    type=str,
    metavar='LOG_FILEPATH',
    help=('Path of an additional log file')
)

def worker(*commands):
    # command = list(commands)
    # pkwargs = {
    #     'stdout': PIPE, 'stderr': STDOUT, 'bufsize': 0
    # }
    # with Popen(command, **pkwargs) as proc, open('test.log', 'w') as log:
    #     for line in proc.stdout:
    #         sys.stdout.buffer.write(line)
    #         log.write(line)
            
    subprocess.run(list(commands)) # [sys.executable, r'spotmax\test.py'])
    # sys.stdout.flush()

if __name__ == '__main__':
    args = vars(ap.parse_args())
    command = args['command']
    commands = command.split(',')
    commands = [command.lstrip() for command in commands]
    process = multiprocessing.Process(target=worker, args=commands)
    process.start()
    process.join()
