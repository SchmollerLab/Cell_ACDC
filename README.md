# <img src="https://github.com/SchmollerLab/Cell_ACDC/blob/main/src/resources/assign-motherbud.svg" width="60" height="60"> Cell-ACDC

### A Python GUI-based framework for <b>segmentation</b>, <b>tracking</b> and <b>cell cycle annotations</b> of microscopy data

Written in Python 3.8 by Francesco Padovani and Benedikt Mairhoermann.

Tested on Windows 10 (64 bit), macOS, and Linux Mint 20.1

*NOTE: This readme is not an exhaustive manual. Please find a <b>User Manual</b> (including detailed installation instructions) [here](https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf).*

## Installation using Anaconda

1. Download the [latest release](https://github.com/SchmollerLab/Cell_ACDC/releases) of Cell-ACDC.
2. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for **Python 3.8**
3. Unzip the Cell-ACDC source code, open a terminal and navigate with `cd` command to the Cell-ACDC folder
4. Update conda with `conda update conda`. Optionally, consider removing unused package with the command `conda clean --all`
5. Install the environment with `conda env create --file acdc.yml`. Creating the environment will take several minutes.

## Installation using Pip

1. Download the [latest release](https://github.com/SchmollerLab/Cell_ACDC/releases) of Cell-ACDC.
2. Download and install [Python 3.8](https://www.python.org/downloads/)
3. Unzip the Cell-ACDC source code, open a terminal and navigate with `cd` command to the Cell-ACDC folder
4. Upgrade pip with `pip install --updgrade pip`
5. Create a virtual environment with `python -m venv env`
6. Install all the dependencies with `pip install -r requirements.txt`

## Running Cell-ACDC

1. Open a terminal and navigate to Cell-ACDC folder
2. Activate the environment (conda: `conda activate acdc`, pip on Windows: `.\env\Scripts\activate`, pip on Unix: `source env/bin/activate`)
3. Navigate to `src` folder and run the main launcher with `python main.py`

## Usage

For details about how to use Cell-ACDC please read the User Manual found in `Cell-ACDC/UserManual` folder.
