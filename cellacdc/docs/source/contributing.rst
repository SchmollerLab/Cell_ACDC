.. _contributing-guide:

How to contribute to Cell-ACDC
==============================

Contributions to Cell-ACDC are **always very welcome**! If you have questions about 
contributing feel free to open a new thread on our 
`forum <https://github.com/SchmollerLab/Cell_ACDC/discussions>`_.

Development process
-------------------

1. If this is the first time you contribute:

   * Go to our `GitHub page <https://github.com/SchmollerLab/Cell_ACDC>`_ 
     and click the "fork" button to create your own copy of the project.

   * Open a terminal window. On Windows I recommend using the `PowerShell 7 
     <https://learn.microsoft.com/en-ie/powershell/scripting/install/installing-powershell-on-windows>`_

   * Clone the forked project to your local computer (remember to replace `your-username` in the link below)::

        git clone https://github.com/your-username/Cell_ACDC.git

   * Navigate to the ``Cell_ACDC`` directory::

        cd Cell_ACDC

   * Add the upstream repository::

        git remote add upstream https://github.com/SchmollerLab/Cell_ACDC.git

   * Now, you have remote repositories named:

     - ``upstream``, which refers to the original ``Cell_ACDC`` repository
     - ``origin``, which refers to your personal fork

   * Install the cloned Cell-ACDC in developer mode (i.e. editable) in a 
     virtual environment using ``venv`` or ``conda``:

     * ``venv`` (more info `here <https://docs.python.org/3/library/venv.html>`_)
  
       ::

         # Navigate to a folder where you want to create the virtual env
         cd ~/.virtualenvs

         # Create a virtual env with the name you like, e.g., ``acdc-dev``
         python -m pip venv acdc-dev

         # Activate the environment (on PowerShell replace ``source`` with dot ``.``)
         source ~/.virtualenvs/acdc-dev/bin/activate

         # Navigate to the cloned folder path (parent folder of ``cellacdc``)
         cd <path_to_Cell_ACDC>

         # Install Cell-ACDC in developer mode
         pip install -e .
  
     * ``conda`` (Miniforge)

       ::

         # Create a virtual env with the name you like, e.g., ``acdc-dev``
         conda create -n acdc-dev python=3.12

         # Activate the environment
         conda activate acdc-dev

         # Navigate to the cloned folder path (parent folder of ``cellacdc``)
         cd <path_to_Cell_ACDC>

         # Install Cell-ACDC in developer mode
         pip install -e .

2. Develop your contribution:

   * Navigate to the cloned folder path (parent folder of ``cellacdc``)::
        
        cd <path_to_Cell_ACDC>
    
   * Pull the latest changes from upstream::

        git checkout main
        git pull upstream main

   * Create a branch with the name you prefer, such as 'new-segm-model'::

        git checkout -b new-segm-model

   * Commit locally as you progress (with ``git add`` and ``git commit``). Please write `good commit messages <https://vxlabs.com/software-development-handbook/#good-commit-messages>`_.

3. Submit your contribution through a Pull Request (PR):

   * Push your changes back to your fork on GitHub::

        git push origin new-segm-model

   * Go to GitHub. The new branch will show up with a green "pull request" button -- click it.
  
Note that if you want to modify the code of the Pull Request, you can simply 
commit and push to the same branch. GitHub will automatically add to the open 
Pull Request.

Where the code lives
--------------------

The file ``CODEMAP.md`` at the root of the repository
(`on GitHub <https://github.com/SchmollerLab/Cell_ACDC/blob/main/CODEMAP.md>`_)
is the map of the ``cellacdc`` package: one line per module, where new code
goes, which module to change for common tasks and where the names of the old
single modules (``gui.py``, ``apps.py``, ``widgets.py``, ``myutils.py``, ...)
went when Cell-ACDC was reorganized into packages. In short, each part of the
program is a package with its own dialogs, widgets and workers
(``launcher``, ``data_structure``, ``data_prep``, ``segmentation_module``,
``gui``, ``measurements_viewer``, ``utils``), code used by several parts is
in ``shared``, and the library modules without a window stay at the top level
of ``cellacdc``. When you add or move a module, run
``python tools/codemap.py`` and describe the new entry in ``CODEMAP.md``
(``tests/test_codemap.py`` checks it).

The old import paths (for example ``cellacdc.apps`` or
``cellacdc.gui_navigation``) still work for code outside Cell-ACDC, with a
``DeprecationWarning`` that names the new path. Cell-ACDC's own code and tests
must use the new paths: the tests fail on the warning.

Running the tests
-----------------

The tests are in the ``tests`` folder and use `pytest <https://docs.pytest.org>`_.
Install pytest and the GUI requirements of your Qt binding in the development
environment, e.g.::

    pip install pytest -r requirements_gui_pyqt6.txt

Then run the tests from the cloned folder (parent folder of ``cellacdc``). The
GUI tests run offscreen, so no windows appear::

    # The fast tier, as CI runs it on every push to a pull request
    python -m pytest -m "(smoke or not slow) and not flaky"

    # The full suite, as CI runs it before merging
    python -m pytest

    # Only the smoke tests (core workflows, less than a minute)
    python -m pytest -m smoke

    # The slow tests of the measurements viewer
    python -m pytest -m "viewer and slow"

    # The tests related to your changes (compared with upstream/main;
    # arguments after --run go to pytest)
    python tests/related_tests.py --base upstream/main --run

The tests are marked by speed and purpose (markers registered in
``pyproject.toml``):

* ``slow``: the test takes half a second or more (a parametrized test: all
  its cases together take 5 seconds or more).
* ``smoke``: the test checks a core workflow, e.g. the GUI or the launcher
  starts, a Position loads, the measurements viewer loads data and plots, a
  configuration is saved and loaded again. Smoke tests run on every push, also
  when they are slow.
* ``flaky``: the test failed or hung now and then on CI (timing, threads).
  Like the slow tests, it is left out of the fast tier until it is fixed.

Each test file also has exactly one area marker, for the part of Cell-ACDC it
tests. It is set for the whole file at the top, e.g.
``pytestmark = pytest.mark.viewer`` (a list when the file has other module
markers such as ``slow``):

* ``viewer``: the measurements viewer and its predictors, phases, equations and
  measurement processing.
* ``gui``: the main GUI, the launcher and shared widgets and dialogs.
* ``models``: segmentation models, trackers, promptable models and the model
  server.
* ``core``: data loading, measurements, image analysis algorithms and batch
  utilities.
* ``infra``: the test setup, packaging, resources and repository tooling.

A new test file needs its area marker (``tests/test_area_markers.py`` fails
without it). When you add a test, keep it fast (small synthetic data) and check
its duration with ``python -m pytest tests/test_<name>.py --durations=0``. Mark
it ``@pytest.mark.slow`` if it still takes half a second or more, or add
``pytest.mark.slow`` to the file's ``pytestmark`` when all its tests are slow. A
test must pass whatever other tests run before it: the tiers leave out the slow
and flaky tests.

Continuous integration (GitHub Actions) runs the tests in tiers, with Python
3.10 to 3.13 on Windows and macOS (PyQt5 and PyQt6) and on Ubuntu (PyQt5):

* **Every push to a pull request**: flake8 and the fast tier (workflows
  ``build-*.yml``, a few minutes), plus, on Ubuntu only, the slow and flaky
  tests of the test files related to the changed files
  (``tests/related_tests.py``).
* **Before merging**: the full suite (workflow ``full-tests.yml``). A maintainer
  adds the ``full tests`` label to the pull request (the workflow looks for
  exactly this label name): the suite runs now and again on every push while
  the label stays (a new push or removing the label cancels a running full
  suite). It can also be started from the
  `Actions tab <https://github.com/SchmollerLab/Cell_ACDC/actions/workflows/full-tests.yml>`_
  ("Run workflow") on any branch that contains ``full-tests.yml`` (merge
  ``main`` into older branches first). Its checks are named ``full / <os>,
  <Qt binding>, Python <version>``; without the label a single skipped check
  ``full`` is listed.
* **After merging**: every push to ``main`` and a weekly run (Monday) run the
  full suite (workflows ``build-*.yml``, shown by the badges of the README).
