.. |acdclogo| image:: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/refs/heads/main/cellacdc/resources/logo_v2.svg
   :width: 80

.. |githublogo| image:: images/github_logo.png
   :width: 32
   :target: https://github.com/SchmollerLab/Cell_ACDC

.. _GitHub: https://github.com/SchmollerLab/Cell_ACDC

|acdclogo| Cell-ACDC
====================

A GUI-based Python framework for **segmentation**, **tracking**, **cell cycle annotations** and **quantification** of microscopy data
-------------------------------------------------------------------------------------------------------------------------------------

|githublogo| Source code on `GitHub`_

*Written in Python 3 by* \ `Francesco Padovani <https://github.com/ElpadoCan>`__ \ *and* \ `Benedikt Mairhoermann <https://github.com/Beno71>`__\ *.*

*Core developers:* `Francesco Padovani <https://github.com/ElpadoCan>`__, `Timon Stegmaier <https://github.com/Teranis>`__, \ *and* \ `Benedikt Mairhoermann <https://github.com/Beno71>`__\ *.*

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

.. tip:: 

    Check out `here <https://youtu.be/u1cQ2MH5uEQ?si=-_hpBoJIccMRjrazs>`__ for a 
    **video tutorial** introdcuing Cell-ACDC. It is a workshop we held at the 
    I2K Conference 2024.

.. carousel::
   :show_captions_below:
   :show_controls:
   :show_indicators:

   .. figure:: images/home_carousel/spheroid_Mario.png

      Sphereoid segmentation

      Segment and quantify the spheroid in 3D
   
   .. figure:: images/home_carousel/yeast_Lisa.png

      Yeast segmentation

      Segment, track, and annotate cell cycle
   
   .. figure:: images/home_carousel/C_elegans_Nada.png

      Nuclei segmentation in *C. elegans*

      Segment sub-set of nuclei in multi-cellular organisms
   
   .. figure:: images/home_carousel/measurments_gui.png

      Compute measurements 

      Easily compute several intesity and morphological measurements
   
   .. figure:: images/home_carousel/acdc_launcher_utilities.png

      Cell-ACDC launcher

      Run batch-processing and utilities from the launcher
   
Contents
--------

.. toctree::
   :maxdepth: 2
   
   overview
   installation
   getting-started
   data-structure
   data-structure-fiji
   contributing
   tooltips
   models
   bdir_corr
   datastruc
   troubleshooting/index
   release-notes
   publications
   resources
   citation

Resources
---------

* `GitHub <https://github.com/SchmollerLab/Cell_ACDC>`_
* `Publication <https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6>`_
* `Forum <https://forum.image.sc/tag/Cell-ACDC>`_