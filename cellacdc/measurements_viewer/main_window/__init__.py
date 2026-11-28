"""The measurements viewer window.

- window.py: class MeasurementsViewerWin with its base classes
  (QMainWindow first, then the Viewer<Topic>GuiElements mixins of this
  package, then MeasurementsViewerUtils, then the shared mixins of
  gui/mixins/), class attributes, __init__, the Qt-named overrides, the
  overrides that call super(), and run().
- <topic>.py: one Viewer<Topic>GuiElements mixin per topic (UI setup,
  configuration and premade templates, opening, loading and saving data,
  navigation, image panel, plotting, columns list, measurement
  processing, phases, time points, custom annotations). Module constants
  live with the mixin that reads them.

Add a new window method to the mixin of its topic. Widget-free logic
goes to measurements_viewer/viewer_utils.py, position_data.py or
annotation_data.py.

This __init__ imports nothing.
"""
