# <img src="https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/resources/icons/assign-motherbud.svg" width="80" height="80"> Cell-ACDC

### A GUI-based Python framework for **segmentation**, **tracking** and **cell cycle annotations** of microscopy data

[![build ubuntu](https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-ubuntu.yml/badge.svg)](https://github.com/SchmollerLab/Cell_ACDC/actions)
[![build macos](https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-macos.yml/badge.svg)](https://github.com/SchmollerLab/Cell_ACDC/actions)
[![build windows](https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-windows.yml/badge.svg)](https://github.com/SchmollerLab/Cell_ACDC/actions)
[![Python version](https://img.shields.io/pypi/pyversions/cellacdc)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-brightgreen)](https://github.com/SchmollerLab/Cell_ACDC/blob/main/LICENSE)
[![pypi version](https://img.shields.io/pypi/v/cellacdc?color=red)](https://pypi.org/project/cellacdc/)
[![repo size](https://img.shields.io/github/repo-size/SchmollerLab/Cell_ACDC)](https://github.com/SchmollerLab/Cell_ACDC)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2021.09.28.462199-informational)](https://www.biorxiv.org/content/10.1101/2021.09.28.462199v2)

<p align="left">
  <img src="https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/resources/figures/Fig1.jpg" width="700" alt><br>
    <em>Overview of pipeline and GUI</em>
</p><br>

Feel free to **ask any question** in our [Discussions area](https://github.com/SchmollerLab/Cell_ACDC/discussions)!

You can check out our pre-print [here](https://www.biorxiv.org/content/10.1101/2021.09.28.462199v2) and Twitter thread [here](https://twitter.com/frank_pado/status/1443957038841794561?s=20).

Written in Python 3 by Francesco Padovani and Benedikt Mairhoermann.

Tested on Windows 10 (64 bit), macOS, and Linux Mint 20.1

*NOTE: This readme is not an exhaustive manual. Please find a **User Manual** (including detailed installation instructions) [here](https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf).*

## Overview

Let's face it, when dealing with segmentation of microscopy data we often do not have time to check that **everything is correct**, because it is a **tedious** and **very time consuming process**. Cell-ACDC comes to the rescue!
We combined the currently **best available neural network models** (such as [YeaZ](https://www.nature.com/articles/s41467-020-19557-4),
[Cellpose](https://www.nature.com/articles/s41592-020-01018-x), [StarDist](https://github.com/stardist/stardist), and [YeastMate](https://github.com/hoerlteam/YeastMate)) and we complemented them with a **fast and intuitive GUI**.

We developed and implemented several smart functionalities such as **real-time continuous tracking**, **automatic propagation** of error correction, and several tools to facilitate manual correction, from simple yet useful **brush** and **eraser** to more complex flood fill (magic wand) and Random Walker segmentation routines.

See below **how it compares** to other popular tools available (*Table 1 of our [pre-print](https://www.biorxiv.org/content/10.1101/2021.09.28.462199v2)*).

<p align="center">
  <img src="https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/resources/figures/Table1.jpg" width="600">
</p>

## Is it only about segmentation?

Of course not! Cell-ACDC automatically computes **several single-cell numerical features** such as cell area and cell volume, plus the mean, max, median, sum and quantiles of any additional fluorescent channel. It even performs background correction, to compute the **protein amount and concentration**.

You can load and analyse single **2D images**, **3D data** (3D z-stacks or 2D images over time) and even **4D data** (3D z-stacks over time).

Finally, we provide Jupyter notebooks to **visualize** and interactively **explore** the data produced.

**Do not hesitate to contact me** here on GitHub (by opening an issue) or directly at my email francesco.padovani@helmholtz-muenchen.de for any problem and/or feedback on how to improve the user experience!

## Update v1.2.4
First release that is finally available on PyPi.

Main new feature: custom trackers! You can now add any tracker you want by implementing a simple tracker class. See the [manual](https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf) at the section "**Adding trackers to the pipeline**".

Additionally, this release includes many UI/UX improvements such as color and style customisation, alongside a light/dark mode switch.

## Update v1.2.3

**NOTE: some users had issues installing the environment with this version. Please see this [issue](https://github.com/SchmollerLab/Cell_ACDC/issues/5) for a possible solution**

This release includes new segmentation models:
- Cellpose v0.8.0 with the models cyto2 and omnipose
- StarDist

## Update v1.2.2

This is the first release with **full macOS support**! Additionally, navigating through time-lapse microscopy data is now up to **10x faster** than previous versions.
More details [here](https://github.com/SchmollerLab/Cell_ACDC/releases/tag/v1.2.2)

## Installation using Anaconda (recommended)

*NOTE: If you don't know what Anaconda is or you are not familiar with it, we recommend reading the detailed installation instructions found in manual [here](https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf).*

1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for **Python 3.9**. *IMPORTANT: For Windows make sure to choose the **64 bit** version*.
2. Update conda with `conda update conda`. Optionally, consider removing unused packages with the command `conda clean --all`
3. Create a virtual environment with the command `conda create -n acdc python=3.9`
4. Activate the environment `conda activate acdc`
5. Install Cell-ACDC with the command `pip install cellacdc`

## Installation using Pip

1. Download and install [Python 3.9](https://www.python.org/downloads/)
2. Upgrade pip with `pip install --updgrade pip`
3. Navigate to a folder where you want to create the virtual environment
4. Create a virtual environment: Windows: `py -m venv acdc`, macOS/Unix `python3 -m venv acdc`
5. Activate the environment: Windows: `.\acdc\Scripts\activate`, macOS/Unix: `source acdc/bin/activate`
6. Install Cell-ACDC with the command `pip install cellacdc`

## Install from source

If you want to contribute or try out experimental features (and, if you have time, maybe report a bug or two :D), you can install the developer version from source as follows:

1. Open a terminal and navigate to a folder where you want to download Cell-ACDC
2. Clone the repo with the command `git clone https://github.com/SchmollerLab/Cell_ACDC.git` (if you are on Windows you need to install `git` first. Install it from [here](https://git-scm.com/download/win))
3. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
4. Update conda with `conda update conda`. Optionally, consider removing unused packages with the command `conda clean --all`
5. Create a new conda environment with the command `conda create -n acdc python=3.8`
6. In the terminal, navigate to the `Cell_ACDC` folder that you cloned before and install Cell-ACDC with the command `pip install -e .`.

## Running Cell-ACDC

1. Open a terminal (on Windows use the Anaconda Prompt if you installed with `conda` otherwise we recommend installing and using the [PowerShell 7](https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.2))
2. Activate the environment (conda: `conda activate acdc`, pip on Windows: `.\env\Scripts\activate`, pip on Unix: `source env/bin/activate`)
3. Run the command `acdc` or `cellacdc`

## Usage

For details about how to use Cell-ACDC please read the User Manual downloadable from [here](https://github.com/SchmollerLab/Cell_ACDC/tree/main/UserManual)
