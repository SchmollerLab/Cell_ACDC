"""Conftest for GUI automation tests."""
# Set environment variables BEFORE any imports from cellacdc
import os
import sys

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyqt5'  # Prefer PyQt5

import logging

logger = logging.getLogger(__name__)

# Properly initialize Cell-ACDC setup before tests run
def pytest_configure(config):
    """Initialize Cell-ACDC on pytest startup - called early during pytest initialization."""
    try:
        # Set up config to avoid interactive prompts and model downloads
        from cellacdc import config as acdc_config
        acdc_config.parser_args['yes'] = True  # Auto-accept prompts
        acdc_config.parser_args['cpModelsDownload'] = False
        acdc_config.parser_args['AllModelsDownload'] = False
        
        logger.info("Initializing Cell-ACDC GUI libraries...")
        from cellacdc._run import _setup_gui_libraries, _setup_numpy
        _setup_gui_libraries(exit_at_end=False)
        
        logger.info("Setting up numpy...")
        _setup_numpy()
        
        logger.info("Cell-ACDC initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Cell-ACDC: {e}")
        import traceback
        traceback.print_exc()
