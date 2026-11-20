"""Qt-free core shared by the "Create data structure" GUI (Module 0), the
"Re-structure image files" GUI and the ``acdc-data`` command line tool.

Layering (lower layers never import higher ones):

1. ``errors``, ``naming``, ``units``, ``callbacks``: exceptions and pure
   helpers (standard library only).
2. ``sources``, ``metadata``: raw-file listing and checks, metadata model,
   defaults, user settings and their validation.
3. ``planning``: what to write where (``PositionJob``), existing positions,
   raw-file moves and renames; ``writers``: metadata.csv/XML, Images
   folders, file moves (imports ``planning``, never the reverse).
4. ``tasks`` (run in a worker process or in-process), ``runner`` and
   ``_worker`` (subprocess transport, standard library only),
   ``backends``, ``convert`` (orchestrator) and ``restructure``.

Every module imports without Qt. Other ``cellacdc`` modules are imported at
module level only as module objects (``from .. import myutils``), never as
``from ..myutils import name``, because ``import cellacdc`` executes the GUI
modules while ``myutils``, ``load`` and ``core`` are still partially
initialised. This package intentionally imports nothing here.
"""
