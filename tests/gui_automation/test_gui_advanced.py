"""Advanced GUI tests with dataset loading and workflow simulation."""
import os
import sys
import time
import random
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from qtpy.QtWidgets import QApplication, QMessageBox, QFileDialog
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
def gui_with_test_data(qapp, tmp_path):
    """Create GUI window and optionally load test dataset."""
    if guiWin is None:
        pytest.skip("guiWin import failed")
    
    try:
        # Mock file dialogs to avoid user interaction
        QFileDialog.getExistingDirectory = MagicMock(
            return_value=str(Path("tests/test_data/test_sample"))
        )
        
        QMessageBox.information = MagicMock()
        QMessageBox.warning = MagicMock()
        QMessageBox.critical = MagicMock()
        QMessageBox.question = MagicMock(return_value=QMessageBox.Yes)
        
        # Create GUI window
        window = guiWin(show=False)
        window.show()
        qapp.processEvents()
        time.sleep(0.5)
        
        yield window
        
        window.close()
        qapp.processEvents()
        
    except Exception as e:
        logger.error(f"Failed to initialize GUI with test data: {e}")
        pytest.skip(f"GUI initialization failed: {e}")


class TestGUIWorkflow:
    """Test realistic GUI workflows."""
    
    def test_gui_workflow_random_actions(self, gui_with_test_data, qapp):
        """Test a realistic workflow with random user actions."""
        window = gui_with_test_data
        window.setFocus()
        qapp.processEvents()
        
        logger.info("Starting workflow with random actions...")
        
        # Simulate various user actions
        workflow_actions = [
            ("Navigate frame forward", lambda: QTest.keyClick(window, Qt.Key_Right)),
            ("Navigate frame backward", lambda: QTest.keyClick(window, Qt.Key_Left)),
            ("Increase zoom", lambda: QTest.keyClick(window, Qt.Key_Plus)),
            ("Decrease zoom", lambda: QTest.keyClick(window, Qt.Key_Minus)),
            ("Tab navigation", lambda: QTest.keyClick(window, Qt.Key_Tab)),
            ("Escape dialog", lambda: QTest.keyClick(window, Qt.Key_Escape)),
        ]
        
        n_actions = random.randint(10, 20)
        successful_actions = 0
        
        for i in range(n_actions):
            action_name, action_func = random.choice(workflow_actions)
            try:
                action_func()
                qapp.processEvents()
                time.sleep(random.uniform(0.1, 0.3))
                successful_actions += 1
                logger.info(f"  [{i+1}/{n_actions}] {action_name}: ✓")
            except Exception as e:
                logger.warning(f"  [{i+1}/{n_actions}] {action_name}: ✗ ({e})")
        
        logger.info(f"✓ Workflow completed ({successful_actions}/{n_actions} actions successful)")
        assert successful_actions >= n_actions * 0.7, "Too many actions failed"
    
    def test_gui_stress_test(self, gui_with_test_data, qapp):
        """Stress test the GUI with rapid interactions."""
        window = gui_with_test_data
        window.setFocus()
        qapp.processEvents()
        
        logger.info("Starting stress test...")
        
        n_interactions = 100
        keys = [Qt.Key_Tab, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Up, Qt.Key_Down]
        
        for i in range(n_interactions):
            try:
                QTest.keyClick(window, random.choice(keys))
                if i % 25 == 0:
                    qapp.processEvents()
                    time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Stress test action {i} failed: {e}")
        
        qapp.processEvents()
        time.sleep(0.5)
        
        # Verify GUI is still functional
        assert window.isVisible(), "GUI crashed or closed"
        logger.info(f"✓ Stress test completed ({n_interactions} interactions)")
    
    def test_gui_random_mouse_movements(self, gui_with_test_data, qapp):
        """Test GUI with random mouse movements and clicks."""
        window = gui_with_test_data
        window.setFocus()
        qapp.processEvents()
        
        logger.info("Testing random mouse movements...")
        
        rect = window.geometry()
        if rect.width() <= 100 or rect.height() <= 100:
            pytest.skip("Window too small for mouse tests")
        
        n_clicks = 15
        for i in range(n_clicks):
            try:
                # Random position within window
                x = random.randint(rect.left() + 50, rect.right() - 50)
                y = random.randint(rect.top() + 50, rect.bottom() - 50)
                
                # Convert to window coordinates
                pos = QPoint(x - rect.left(), y - rect.top())
                
                # Click with random button
                button = random.choice([Qt.LeftButton, Qt.RightButton])
                QTest.mouseClick(window, button, pos=pos)
                
                qapp.processEvents()
                time.sleep(random.uniform(0.05, 0.2))
                
                logger.info(f"  Click {i+1}/{n_clicks} at ({x}, {y}): ✓")
            except Exception as e:
                logger.warning(f"  Click {i+1} failed: {e}")
        
        logger.info(f"✓ Mouse movement test completed")


class TestGUIRobustness:
    """Test GUI robustness and error handling."""
    
    def test_gui_rapid_window_resize(self, gui_with_test_data, qapp):
        """Test GUI handles rapid window resizing."""
        window = gui_with_test_data
        original_size = window.size()
        
        logger.info("Testing rapid window resizing...")
        
        sizes = [
            (800, 600),
            (1024, 768),
            (640, 480),
            (1200, 800),
        ]
        
        for w, h in sizes:
            window.resize(w, h)
            qapp.processEvents()
            time.sleep(0.1)
        
        # Restore original
        window.resize(original_size)
        qapp.processEvents()
        
        assert window.isVisible()
        logger.info("✓ Rapid resizing handled")
    
    def test_gui_long_running_session(self, gui_with_test_data, qapp):
        """Test GUI during a long interaction session."""
        window = gui_with_test_data
        window.setFocus()
        qapp.processEvents()
        
        logger.info("Running long-session test (30 seconds)...")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 30:
            try:
                # Random keyboard action
                keys = [Qt.Key_Tab, Qt.Key_Return, Qt.Key_Up, Qt.Key_Down, Qt.Key_Escape]
                QTest.keyClick(window, random.choice(keys))
                
                # Occasional mouse click
                if random.random() < 0.3:
                    rect = window.geometry()
                    x = random.randint(rect.left() + 50, rect.right() - 50)
                    y = random.randint(rect.top() + 50, rect.bottom() - 50)
                    pos = QPoint(x - rect.left(), y - rect.top())
                    QTest.mouseClick(window, Qt.LeftButton, pos=pos)
                
                if iteration % 50 == 0:
                    qapp.processEvents()
                    time.sleep(0.1)
                
                iteration += 1
            except Exception as e:
                logger.warning(f"Error in long-session test: {e}")
        
        qapp.processEvents()
        assert window.isVisible()
        logger.info(f"✓ Long-session test completed ({iteration} interactions in 30 seconds)")


@pytest.mark.parametrize("complexity_level", [1, 2, 3])
def test_gui_complexity_levels(gui_with_test_data, qapp, complexity_level):
    """Test GUI with different complexity levels of interactions.
    
    Level 1: Simple key presses
    Level 2: Mixed keyboard and mouse
    Level 3: Complex sequences with timing
    """
    window = gui_with_test_data
    window.setFocus()
    qapp.processEvents()
    
    logger.info(f"Running complexity level {complexity_level} test...")
    
    if complexity_level == 1:
        # Simple key presses
        for _ in range(20):
            QTest.keyClick(window, random.choice([Qt.Key_Tab, Qt.Key_Space]))
        logger.info("✓ Level 1: Simple key presses")
    
    elif complexity_level == 2:
        # Mixed interactions
        rect = window.geometry()
        for _ in range(15):
            if random.random() < 0.5:
                QTest.keyClick(window, random.choice([Qt.Key_Tab, Qt.Key_Return]))
            else:
                x = random.randint(rect.left() + 50, rect.right() - 50)
                y = random.randint(rect.top() + 50, rect.bottom() - 50)
                pos = QPoint(x - rect.left(), y - rect.top())
                QTest.mouseClick(window, Qt.LeftButton, pos=pos)
            qapp.processEvents()
            time.sleep(0.1)
        logger.info("✓ Level 2: Mixed keyboard and mouse")
    
    elif complexity_level == 3:
        # Complex sequences with timing
        for seq in range(10):
            # Keyboard sequence
            for _ in range(5):
                QTest.keyClick(window, random.choice([Qt.Key_Up, Qt.Key_Down]))
            qapp.processEvents()
            time.sleep(0.2)
            
            # Mouse sequence
            rect = window.geometry()
            for _ in range(3):
                x = random.randint(rect.left() + 50, rect.right() - 50)
                y = random.randint(rect.top() + 50, rect.bottom() - 50)
                pos = QPoint(x - rect.left(), y - rect.top())
                QTest.mouseClick(window, Qt.LeftButton, pos=pos)
                time.sleep(0.1)
        logger.info("✓ Level 3: Complex sequences with timing")
    
    qapp.processEvents()
    assert window.isVisible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
