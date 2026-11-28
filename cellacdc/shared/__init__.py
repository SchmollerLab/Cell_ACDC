"""Code used by more than one program part.

The program parts are launcher/, data_structure/, data_prep/,
segmentation_module/, gui/, measurements_viewer/ and the batch utilities
in utils/. A dialog, widget or worker belongs here when two or more
parts use it, or a library module (load.py, core.py, prompts.py,
plot.py, myutils/, ...), a plugin (models/, trackers/,
promptable_models/, measurement_processing/) or spotMAX. Code with a
single user part lives in that part's package. Cohesive families stay
together even when some members have one user (the PushButton family in
widgets/push_buttons.py).

- dialogs/: shared dialogs, by theme.
- widgets/: shared widgets and pyqtgraph items, by theme, and the style
  module.
- workers/: the worker infrastructure and the shared workers.

Modules here never import a program part package at module level. This
package re-exports nothing: import its modules.
"""
