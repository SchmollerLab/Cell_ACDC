"""Dialogs used only by the measurements viewer, one module per theme: the
editors of time point identifiers, predictors and filters (with the
custom annotation predictor group box and disabled_while_nested), the
premade template setup, conditional formatting, the phase translator,
the measurement processing recipe.

The predictor editors are also reached from the custom annotation dialog
in gui/mixins/dialogs.py, through a lazy import made only when the
viewer passes predictor options: an accepted exception to the placement
rules. Shared bases (the combine-measurements editors, the preprocessing
widgets) live in shared/dialogs/.

This package re-exports nothing: import its modules.
"""
