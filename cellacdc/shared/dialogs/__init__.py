"""Dialogs used by several program parts, library modules or plugins, one
module per theme: generic input, list, tree and progress dialogs; file,
folder and Position selection; image metadata; image tools (slideshow,
z-crop); Set measurements; the combine-measurements equation editors;
preprocessing and function parameters; segmentation model selection and
parameters; tracker parameters.

base.py holds the QBaseDialog subclass the dialogs derive from. It is a
foundation leaf: it imports no other dialog or widget module, so every
dialog module can import it at module level.

A dialog used by one program part lives in that part (<part>/dialogs.py
or <part>/dialogs/), one opened only from the GUI mixins shared with the
measurements viewer in gui/mixins/dialogs.py, one used by one batch
utility in utils/<utility>.py and one used by several utilities in
utils/base.py.

This package re-exports nothing: import its modules.
"""
