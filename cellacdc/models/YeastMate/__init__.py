import os
import sys
import subprocess

yeastmate_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, yeastmate_path)

# Check missing packages
try:
    import fvcore
except ModuleNotFoundError:
    subprocess.run('pip install fvcore==0.1.5.post20210924', shell=True)

try:
    import omegaconf
except ModuleNotFoundError:
    subprocess.run('pip install omegaconf==2.1.1', shell=True)

try:
    import torchvision
except ModuleNotFoundError:
    subprocess.run('pip install torchvision==0.10.0', shell=True)

try:
    import cloudpickle
except ModuleNotFoundError:
    subprocess.run('pip install cloudpickle==2.0.0', shell=True)
