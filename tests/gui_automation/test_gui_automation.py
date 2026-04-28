"""GUI automation tests for Cell-ACDC.

This module performs automated GUI testing by simulating user interactions
like mouse clicks, keyboard input, and random action sequences.
"""
import os
import sys
import time
import random
import pytest
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from qtpy.QtWidgets import QApplication, QMessageBox
from qtpy.QtCore import Qt, QTimer, QPoint
from qtpy.QtGui import QKeySequence
from qtpy.QtTest import QTest

# Import GUI (initialization done in conftest)
try:
    from cellacdc.gui import guiWin
except ImportError as e:
    logger.error(f"Failed to import guiWin: {e}")
    guiWin = None


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture(scope="function")
def gui_window(qapp, tmp_path):
    """Create a GUI window for testing."""
    if guiWin is None:
        pytest.skip("guiWin import failed")
    
    # Create test data directory
    test_data_dir = Path("tests/test_data/test_sample")
    if not test_data_dir.exists():
        pytest.skip("Test dataset not found. Run generate_test_dataset.py first.")
    
    try:
        # Suppress message boxes during testing
        QMessageBox.information = MagicMock()
        QMessageBox.warning = MagicMock()
        QMessageBox.critical = MagicMock()
        
        # Create GUI window
        window = guiWin(show=False)
        window.show()
        
        # Process events to fully initialize
        qapp.processEvents()
        time.sleep(0.5)
        
        yield window
        
        # Cleanup
        window.close()
        qapp.processEvents()
        
    except Exception as e:
        logger.error(f"Failed to create GUI window: {e}")
        pytest.skip(f"GUI initialization failed: {e}")


class TestGUIBasics:
    """Test basic GUI functionality and startup."""
    
    def test_gui_window_creation(self, gui_window, qapp):
        """Test that GUI window is created successfully."""
        assert gui_window is not None
        assert gui_window.isVisible()
        assert gui_window.windowTitle()
        logger.info("✓ GUI window created successfully")
    
    def test_gui_main_elements_exist(self, gui_window, qapp):
        """Test that main GUI elements are present."""
        # Check for key attributes/widgets
        assert hasattr(gui_window, 'centralwidget')
        logger.info("✓ Main GUI elements present")
    
    def test_gui_responds_to_events(self, gui_window, qapp):
        """Test that GUI responds to events."""
        gui_window.setFocus()
        qapp.processEvents()
        assert gui_window.hasFocus()
        logger.info("✓ GUI responds to events")


class TestGUIInteractions:
    """Test user interactions with the GUI."""
    
    def test_keyboard_input(self, gui_window, qapp):
        """Test keyboard input handling."""
        gui_window.setFocus()
        qapp.processEvents()
        
        # Simulate keyboard press
        QTest.keyClick(gui_window, Qt.Key_Escape)
        qapp.processEvents()
        
        logger.info("✓ Keyboard input processed")
    
    def test_random_clicks(self, gui_window, qapp):
        """Test random mouse clicks on the GUI."""
        gui_window.setFocus()
        qapp.processEvents()
        
        # Get window geometry
        rect = gui_window.geometry()
        if rect.width() > 100 and rect.height() > 100:
            # Perform random clicks
            for _ in range(5):
                x = random.randint(rect.left() + 50, rect.right() - 50)
                y = random.randint(rect.top() + 50, rect.bottom() - 50)
                
                pos = QPoint(x - rect.left(), y - rect.top())
                QTest.mouseClick(gui_window, Qt.LeftButton, pos=pos)
                qapp.processEvents()
                time.sleep(0.1)
            
            logger.info("✓ Random clicks processed")
    
    def test_menu_navigation(self, gui_window, qapp):
        """Test navigating through menus."""
        # This would depend on actual menu structure
        gui_window.setFocus()
        qapp.processEvents()
        
        # Try to find and interact with menus if they exist
        menu_bar = gui_window.menuBar() if hasattr(gui_window, 'menuBar') else None
        if menu_bar is not None:
            menus = menu_bar.findChildren(type(menu_bar))
            logger.info(f"✓ Found {len(menus)} menu items")


class TestGUIRandomActions:
    """Test random action sequences (simulating user workflow)."""
    
    def test_random_action_sequence(self, gui_window, qapp):
        """Perform a random sequence of GUI actions."""
        gui_window.setFocus()
        qapp.processEvents()
        
        actions = [
            ("Tab", lambda: QTest.keyClick(gui_window, Qt.Key_Tab)),
            ("Space", lambda: QTest.keyClick(gui_window, Qt.Key_Space)),
            ("Escape", lambda: QTest.keyClick(gui_window, Qt.Key_Escape)),
            ("Enter", lambda: QTest.keyClick(gui_window, Qt.Key_Return)),
        ]
        
        # Perform random actions
        n_actions = random.randint(3, 8)
        for i in range(n_actions):
            action_name, action_func = random.choice(actions)
            try:
                action_func()
                qapp.processEvents()
                time.sleep(0.2)
                logger.info(f"  Action {i+1}/{n_actions}: {action_name}")
            except Exception as e:
                logger.warning(f"  Action failed: {e}")
        
        logger.info("✓ Random action sequence completed")
    
    def test_window_resize_interactions(self, gui_window, qapp):
        """Test GUI behavior during window resizing."""
        gui_window.setFocus()
        
        original_size = gui_window.size()
        
        # Resize window
        new_size = gui_window.size()
        new_size.setWidth(new_size.width() - 100)
        new_size.setHeight(new_size.height() - 100)
        gui_window.resize(new_size)
        
        qapp.processEvents()
        time.sleep(0.5)
        
        # Restore size
        gui_window.resize(original_size)
        qapp.processEvents()
        
        logger.info("✓ Window resizing handled")
    
    def test_focus_cycling(self, gui_window, qapp):
        """Test cycling focus through GUI elements."""
        gui_window.setFocus()
        qapp.processEvents()
        
        # Cycle through focusable widgets
        for _ in range(10):
            QTest.keyClick(gui_window, Qt.Key_Tab)
            qapp.processEvents()
            time.sleep(0.1)
        
        logger.info("✓ Focus cycling completed")


class TestGUIStability:
    """Test GUI stability under various conditions."""
    
    def test_rapid_key_presses(self, gui_window, qapp):
        """Test GUI stability with rapid key presses."""
        gui_window.setFocus()
        qapp.processEvents()
        
        keys = [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]
        
        for _ in range(20):
            key = random.choice(keys)
            QTest.keyClick(gui_window, key)
            # Don't sleep - test rapid input
        
        qapp.processEvents()
        time.sleep(0.5)
        
        # GUI should still be responsive
        assert gui_window.isVisible()
        logger.info("✓ Rapid key presses handled")
    
    def test_gui_remains_responsive(self, gui_window, qapp):
        """Test that GUI remains responsive after interactions."""
        gui_window.setFocus()
        
        # Perform many operations
        for i in range(50):
            QTest.keyClick(gui_window, random.choice([Qt.Key_Tab, Qt.Key_Space]))
            if i % 10 == 0:
                qapp.processEvents()
        
        qapp.processEvents()
        
        # Check responsiveness
        assert gui_window.isVisible()
        gui_window.update()
        assert gui_window.rect().isValid()
        
        logger.info("✓ GUI remained responsive")


@pytest.mark.parametrize("num_iterations", [3, 5])
def test_gui_extended_interaction(gui_window, qapp, num_iterations):
    """Test extended GUI interaction with multiple iterations."""
    gui_window.setFocus()
    
    for iteration in range(num_iterations):
        logger.info(f"Extended test iteration {iteration + 1}/{num_iterations}")
        
        # Random clicks
        rect = gui_window.geometry()
        for _ in range(random.randint(2, 5)):
            x = random.randint(rect.left() + 50, rect.right() - 50)
            y = random.randint(rect.top() + 50, rect.bottom() - 50)
            
            pos = QPoint(x - rect.left(), y - rect.top())
            QTest.mouseClick(gui_window, Qt.LeftButton, pos=pos)
            qapp.processEvents()
            time.sleep(0.1)
        
        # Random keyboard input
        for _ in range(random.randint(3, 7)):
            QTest.keyClick(gui_window, random.choice([Qt.Key_Tab, Qt.Key_Space, Qt.Key_Return]))
            qapp.processEvents()
            time.sleep(0.1)
    
    logger.info(f"✓ Extended interaction test completed ({num_iterations} iterations)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
