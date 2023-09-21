Installation
============
This guide contains instructions for installing Cell-ACDC in different ways. In the future a video will be added.

.. contents::


Installation using Anaconda (recommended)
-----------------------------------------

1. Install `Anaconda <https://www.anaconda.com/download>`__ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install>`__ (make sure to use the 64-bit version for Windows)
    Anaconda is a distribution of Python, which helps with package management and simplifies many aspects of using Python. Python is a programming language and needs an interpreter to function. Miniconda is a lightweight implementation of Anaconda.
2. Open the terminal (for Windows use the anaconda terminal)
    A terminal is a text-based way to communicate with the computer. The anaconda terminal is a terminal specifically for anaconda, with some commands specific to anaconda only available there.
3. Update conda by typing ``conda update conda``
    This will update all programmes (or packages) that are part of conda
4. Create a virtual environment by typing ``conda create -n acdc python=3.9``
    This will create a virtual environment, which is an isolated instance and partially independent from the rest of the system. The virtual environment is called ``acdc`` in this case.
5. Activate the virtual environment by typing ``conda activate acdc``
    This will activate the environment, meaning that the console is now not in the default system environment, but the ``acdc`` environment. If the activation of the environment was successful, this should be indicated to the left of the active path, circled in red below.

.. image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/docs/source/images/Cmdprompt.png?raw=true
    :target: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/docs/source/images/Cmdprompt.png
    :alt: The active environment is displayed to the left of the currently active path
    :width: 600

6. Update pip using ``python -m pip install --upgrade pip``
    Pip is an application which is included in Python. It manages programmes and updates it. In this case, we tell pip to update itself.
7. Install Cell-ACDC using ``pip install "cellacdc[gui]"``
    This tells pip to install Cell-ACDC, specifically the version with a user interface.

Advanced install methods
------------------------

The following part of the guide is not finished yet and is intended for users who are interested in changing Cell-ACDC's code, thus not really needing the guide.

IMPORTANT: Before installing with other methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are **new to Python** or you need a **refresher** on how to
manage scientific Python environments, I highly recommend reading `this
guide <https://focalplane.biologists.com/2022/12/08/managing-scientific-python-environments-using-conda-mamba-and-friends/>`__
by Dr. Robert Haase BEFORE proceeding with Cell-ACDC installation.

Installation using Pip
~~~~~~~~~~~~~~~~~~~~~~
**Windows**

1. Download and install `Python 3.9 <https://www.python.org/downloads/>`__
2. Open a terminal. On Windows we recommend using the PowerShell that
   you can install from
   `here <https://docs.microsoft.com/it-it/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.2#installing-the-msi-package>`__.
   On macOS use the Terminal app.
3. Upgrade pip: Windows: ``py -m pip install --updgrade pip``,
   macOS/Unix: ``python3 -m pip install --updgrade pip``
4. Navigate to a folder where you want to create the virtual environment
5. Create a virtual environment: Windows: ``py -m venv acdc``,
   macOS/Unix ``python3 -m venv acdc``
6. Activate the environment: Windows: ``.\acdc\Scripts\activate``,
   macOS/Unix: ``source acdc/bin/activate``
7. Install Cell-ACDC with the command ``pip install "cellacdc[gui]"``.
   Note that if you know you are going to **need tensorflow** (for
   segmentation models like YeaZ) you can run the command
   ``pip install "cellacdc[all]"``, or ``pip install tensorflow`` before
   or after installing Cell-ACDC.

Install from source
~~~~~~~~~~~~~~~~~~~

If you want to try out experimental features (and, if you have time,
maybe report a bug or two :D), you can install the developer version
from source as follows:

1.  Install `Anaconda <https://www.anaconda.com/products/individual>`__
    or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__.
2.  Open a terminal and navigate to a folder where you want to download
    Cell-ACDC. If you are on Windows you need to use the “Anaconda
    Prompt” as a terminal. You should find it by searching for “Anaconda
    Prompt” in the Start menu.
3.  Clone the source code with the command
    ``git clone https://github.com/SchmollerLab/Cell_ACDC.git``. If you
    are on Windows you might need to install ``git`` first. Install it
    from `here <https://git-scm.com/download/win>`__.
4.  Navigate to the ``Cell_ACDC`` folder with the command
    ``cd Cell_ACDC``.
5.  Update conda with ``conda update conda``. Optionally, consider
    removing unused packages with the command ``conda clean --all``
6.  Create a new conda environment with the command
    ``conda create -n acdc_dev python=3.9``
7.  Activate the environment with the command
    ``conda activate acdc_dev``
8.  Upgrade pip with the command ``python -m pip install --upgrade pip``
9.  Install Cell-ACDC with the command ``pip install -e .``. The ``.``
    at the end of the command means that you want to install from the
    current folder in the terminal. This must be the ``Cell_ACDC``
    folder that you cloned before.
10. OPTIONAL: If you need tensorflow run the command
    ``pip install tensorflow``.

**Updating Cell-ACDC installed from source**

To update Cell-ACDC installed from source, open a terminal window,
navigate to the Cell_ACDC folder and run the command

::

   git pull

Since you installed with the ``-e`` flag, pulling with ``git`` is
enough.

Install from source with forking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute to the code or you want to have a developer
version that is fixed in time (easier to get back to in case we release
a bug :D) we recommend forking before cloning:

1.  Install `Anaconda <https://www.anaconda.com/products/individual>`__
    or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__.
2.  Create a personal `GitHub account <https://github.com>`__ and log
    in.
3.  Go to the Cell-ACDC `GitHub page <https://github.com/SchmollerLab/Cell_ACDC>`__ and click the
    “Fork” button (top-right) to create your own copy of the project.
4.  Open a terminal and navigate to a folder where you want to download
    Cell-ACDC. If you are on Windows you need to use the “Anaconda
    Prompt” as a terminal. You should find it by searching for “Anaconda
    Prompt” in the Start menu.
5.  Clone the forked repo with the command
    ``git clone https://github.com/your-username/Cell_ACDC.git``.
    Remember to replace the ``your-username`` in the command. If you are
    on Windows you might need to install ``git`` first. Install it from
    `here <https://git-scm.com/download/win>`__.
6.  Navigate to the ``Cell_ACDC`` folder with the command
    ``cd Cell_ACDC``.
7.  Add the upstream repository with the command
    ``git remote add upstream https://github.com/SchmollerLab/Cell_ACDC.git``
8.  Update conda with ``conda update conda``. Optionally, consider
    removing unused packages with the command ``conda clean --all``
9.  Create a new conda environment with the command
    ``conda create -n acdc_dev python=3.9``. Note that ``acdc_dev`` is
    the name of the environment and you can call it whatever you like.
    Feel free to call it just ``acdc``.
10. Activate the environment with the command
    ``conda activate acdc_dev``
11. Upgrade pip with the command ``python -m pip install --upgrade pip``
12. Install Cell-ACDC with the command ``pip install -e .``. The ``.``
    at the end of the command means that you want to install from the
    current folder in the terminal. This must be the ``Cell_ACDC``
    folder that you cloned before.
13. OPTIONAL: If you need tensorflow run the command
    ``pip install tensorflow``.

**Updating Cell-ACDC installed from source with forking**

To update Cell-ACDC installed from source, open a terminal window,
navigate to the Cell-ACDC folder and run the command

::

   git pull upstream main

Since you installed with the ``-e`` flag, pulling with ``git`` is
enough.