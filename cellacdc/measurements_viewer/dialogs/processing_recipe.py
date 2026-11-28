"""Recipe dialog, parameter widget and function selector of measurement
processing in the measurements viewer (plugins:
measurement_processing/).
"""
from cellacdc.shared.widgets import combo_boxes as widgets_combo_boxes


class SignalProcessingSelector(widgets_combo_boxes.PreProcessingSelector):
    from cellacdc.measurement_processing import MEASUREMENT_PROCESSING_MAPPER as preprocessMapper
