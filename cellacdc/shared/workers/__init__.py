"""Workers used by several program parts.

- base.py: the worker infrastructure (worker_exception_handler,
  workerLogger, signals) and the generic workers (SimpleWorker,
  CustomPreprocessWorkerGUI, ...). A foundation leaf: it imports no
  dialog, widget, gui or utils module, so every worker module can import
  it at module level.
- saving.py: saveDataWorker, the save worker of the main GUI, also used
  by the Compute measurements utility.

Workers of one program part live in that part: <part>/workers.py, for
the main GUI gui/workers/<theme>.py. Utility workers live in their
utils/<utility>.py, BaseWorkerUtil in utils/base.py, and the workers
whose names clash with utility window classes in utils/segm_workers.py.

This package re-exports nothing: import its modules.
"""
