.. |acdclogo| image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/6bf8442b6a33d41fa9de09a2098c6c2b9efbcff1/cellacdc/resources/logo.svg
   :width: 80

|acdclogo| Welcome to Cell-ACDC!
================================

A GUI-based Python framework for **segmentation**, **tracking**, **cell cycle annotations** and **quantification** of microscopy data
-------------------------------------------------------------------------------------------------------------------------------------

*Written in Python 3 by* \ `Francesco Padovani <https://github.com/ElpadoCan>`__ \ *and* \ `Benedikt Mairhoermann <https://github.com/Beno71>`__\ *.*

.. |build_win_pyqt5| image:: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-windows_pyqt5.yml/badge.svg
   :target: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-windows_pyqt5.yml
   :alt: Build Status (Windows PyQt5)

.. |build_ubuntu_pyqt5| image:: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-ubuntu_pyqt5.yml/badge.svg
   :target: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-ubuntu_pyqt5.yml
   :alt: Build Status (Ubuntu PyQt5)

.. |build_macos_pyqt5| image:: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-macos_pyqt5.yml/badge.svg
   :target: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-macos_pyqt5.yml
   :alt: Build Status (macOS PyQt5)

.. |build_win_pyqt6| image:: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-windows_pyqt6.yml/badge.svg
   :target: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-windows_pyqt6.yml
   :alt: Build Status (Windows PyQt6)

.. |build_macos_pyqt6| image:: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-macos_pyqt6.yml/badge.svg
   :target: https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/build-macos_pyqt6.yml
   :alt: Build Status (macOS PyQt6)

.. |py_version| image:: https://img.shields.io/pypi/pyversions/cellacdc
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. |pypi_version| image:: https://img.shields.io/pypi/v/cellacdc?color=red
   :target: https://pypi.org/project/cellacdc/
   :alt: PyPi Version

.. |downloads_month| image:: https://static.pepy.tech/badge/cellacdc/month
   :target: https://pepy.tech/project/cellacdc
   :alt: Downloads per month

.. |license| image:: https://img.shields.io/badge/license-BSD%203--Clause-brightgreen
   :target: https://github.com/SchmollerLab/Cell_ACDC/blob/main/LICENSE
   :alt: License

.. |repo_size| image:: https://img.shields.io/github/repo-size/SchmollerLab/Cell_ACDC
   :target: https://github.com/SchmollerLab/Cell_ACDC
   :alt: Repository Size

.. |doi| image:: https://img.shields.io/badge/DOI-10.1101%2F2021.09.28.462199-informational
   :target: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6
   :alt: DOI

.. |docs| image:: https://readthedocs.org/projects/cell-acdc/badge/?version=latest
    :target: https://cell-acdc.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

|build_win_pyqt5| |build_ubuntu_pyqt5| |build_macos_pyqt5| |build_win_pyqt6|
|build_macos_pyqt6| |py_version| |pypi_version| |downloads_month| |license|
|repo_size| |doi| |docs|

|

.. image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/figures/Fig1.jpg
   :alt: Overview of pipeline and GUI
   :width: 600

Overview of pipeline and GUI

Overview
========
Let's face it, when dealing with segmentation of microscopy data we
often do not have time to check that **everything is correct**, because
it is a **tedious** and **very time consuming process**. Cell-ACDC comes
to the rescue! We combined the currently **best available neural network
models** (such as `Segment Anything Model
(SAM) <https://github.com/facebookresearch/segment-anything>`__,
`YeaZ <https://www.nature.com/articles/s41467-020-19557-4>`__,
`cellpose <https://www.nature.com/articles/s41592-020-01018-x>`__,
`StarDist <https://github.com/stardist/stardist>`__,
`YeastMate <https://github.com/hoerlteam/YeastMate>`__,
`omnipose <https://omnipose.readthedocs.io/>`__,
`delta <https://gitlab.com/dunloplab/delta>`__,
`DeepSea <https://doi.org/10.1016/j.crmeth.2023.100500>`__, etc.) and we
complemented them with a **fast and intuitive GUI**.

We developed and implemented several smart functionalities such as
**real-time continuous tracking**, **automatic propagation** of error
correction, and several tools to facilitate manual correction, from
simple yet useful **brush** and **eraser** to more complex flood fill
(magic wand) and Random Walker segmentation routines.

See below **how it compares** to other popular tools available (*Table 1
of
our* \ `publication <https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6>`__).

.. image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/figures/Table1.jpg
  :width: 700

Is it only about segmentation?
------------------------------

Of course not! Cell-ACDC automatically computes **several single-cell
numerical features** such as cell area and cell volume, plus the mean,
max, median, sum and quantiles of any additional fluorescent channel's
signal. It even performs background correction, to compute the **protein
amount and concentration**.

You can load and analyse single **2D images**, **3D data** (3D z-stacks
or 2D images over time) and even **4D data** (3D z-stacks over time).

Finally, we provide Jupyter notebooks to **visualize** and interactively
**explore** the data produced.

Bidirectional microscopy shift error correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Is every second line in your files from your bidirectional microscopy
shifted? Look
`here <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/scripts/README.md>`__
for further information on how to correct your data.

Resources
=========
- Please find a complete user guide `here <https://cell-acdc.readthedocs.io/en/latest/>`__
- `Installation guide <https://cell-acdc.readthedocs.io/en/latest/installation.html#installation-using-anaconda-recommended>`__
- `User manual <https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf>`__
- `Publication <https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6>`__ of Cell-ACDC
- `Forum <https://github.com/SchmollerLab/Cell_ACDC/discussions>`__ for discussions (feel free to **ask any question**)
- **Report issues, request a feature or ask questions** by opening a new issue `here <https://github.com/SchmollerLab/Cell_ACDC/issues>`__
- X `thread <https://twitter.com/frank_pado/status/1443957038841794561?s=20>`__

Citing Cell-ACDC and the available models
=========================================

If you find Cell-ACDC useful, please cite it as follows:

   Padovani, F., Mairhörmann, B., Falter-Braun, P., Lengefeld, J. & 
   Schmoller, K. M. Segmentation, tracking and cell cycle analysis of live-cell 
   imaging data with Cell-ACDC. *BMC Biology* 20, 174 (2022). 
   DOI: `10.1186/s12915-022-01372-6 <https://doi.org/10.1186/s12915-022-01372-6>`_ 

**IMPORTANT**: when citing Cell-ACDC make sure to also cite the paper of the 
segmentation models and trackers you used! 
See `here <https://cell-acdc.readthedocs.io/en/latest/citation.html>`_ for a list of models currently available in Cell-ACDC.

Contact
=======
**Do not hesitate to contact us** here on GitHub (by opening an issue)
or directly at the email padovaf@tcd.ie for any problem and/or feedback
on how to improve the user experience!

Contributing
============

At Cell-ACDC we encourage contributions to the code! Please read our 
`contributing guide <https://github.com/SchmollerLab/Cell_ACDC/blob/main/cellacdc/docs/source/contributing.rst>`_ 
to get started.