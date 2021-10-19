import os
import sys
import subprocess

yeastmate_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, yeastmate_path)

# Check missing packages
try:
    import pycocotools
except ModuleNotFoundError:
    subprocess.run('pip install pycocotools==2.0.2', shell=True)

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
    import pycocotools
except ModuleNotFoundError:
    subprocess.run('pip install pycocotools==2.0.2', shell=True)
