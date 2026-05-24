#!/usr/bin/env python
import os
import logging

import os
import numpy as np

import site

sitepackages = site.getsitepackages()
site_packages = [p for p in sitepackages if p.endswith("site-packages")][0]

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
cellacdc_installation_path = os.path.dirname(cellacdc_path)
if cellacdc_installation_path != site_packages:
    # Running developer version. Delete cellacdc folder from site_packages
    # if present from a previous installation of cellacdc from PyPi
    cellacdc_path_pypi = os.path.join(site_packages, "cellacdc")
    if os.path.exists(cellacdc_path_pypi):
        import shutil

        try:
            shutil.rmtree(cellacdc_path_pypi)
        except Exception as err:
            print(err)
            print(
                "[ERROR]: Previous Cell-ACDC installation detected. "
                f"Please, manually delete this folder and re-start the software "
                f'"{cellacdc_path_pypi}". '
                "Thank you for you patience!"
            )
            exit()
        print("*" * 60)
        input(
            "[WARNING]: Cell-ACDC had to clean-up and older installation. "
            "Please, re-start the software. Thank you for your patience! "
            "(Press any key to exit). "
        )
        exit()

from cellacdc import _run


def run():
    from cellacdc.config import parser_args

    PARAMS_PATH = parser_args["params"]

    if parser_args["version"] or parser_args["info"]:
        from cellacdc.utils import get_info_version_text

        info_txt = get_info_version_text()
        print(info_txt)
        exit()

    if parser_args["reset"]:
        from cellacdc.utils import reset_settings

        reset_info_txt = reset_settings()
        print(reset_info_txt)
        exit()

    if PARAMS_PATH:
        _run.run_cli(PARAMS_PATH)
    else:
        run_gui()


def main():
    # Keep compatibility with users that installed older versions
    # where the entry point was main()
    run()


def run_gui():
    app, splashScreen = _run.setup_gui_runtime(splashscreen=True)

    from cellacdc import utils, printl

    print("Launching application...")

    from cellacdc._main import mainWin

    if not splashScreen.isVisible():
        splashScreen.show()

    win = mainWin(app)

    try:
        utils.check_matplotlib_version(qparent=win)
    except Exception as e:
        pass
    version, success = utils.read_version(logger=win.logger.info, return_success=True)
    if not success:
        error = utils.check_install_package(
            "setuptools_scm", pypi_name="setuptools-scm"
        )
        if error:
            win.logger.info(error)
        else:
            version = utils.read_version(logger=win.logger.info)
    win.setVersion(version)
    win.launchWelcomeGuide()
    win.show()
    try:
        win.welcomeGuide.showPage(win.welcomeGuide.welcomeItem)
    except AttributeError:
        pass
    win.logger.info("**********************************************")
    win.logger.info(f"Welcome to Cell-ACDC v{version}")
    win.logger.info("**********************************************")
    win.logger.info("----------------------------------------------")
    win.logger.info(
        "NOTE: If application is not visible, it is probably minimized\n"
        "or behind some other open windows."
    )
    win.logger.info("----------------------------------------------")
    splashScreen.close()
    # splashScreenApp.quit()
    # modernWin.show()
    app.exec_()
