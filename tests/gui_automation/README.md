# GUI Automation Testing

This directory contains automated GUI tests for the Cell-ACDC application.

## Overview

GUI testing in Cell-ACDC uses a **subprocess-based approach** because:

1. **Proper Initialization**: Tests run Cell-ACDC through the normal entry point (`__main__.py`), ensuring all patches and setup code (like DLL error handling on Windows) are applied correctly.

2. **Avoids Circular Imports**: Cell-ACDC has circular dependencies in its module structure. Running in separate processes avoids import-time issues.

3. **Isolates State**: Each test runs in its own Python process, preventing state contamination between tests.

## Test Structure

### `test_gui_subprocess.py` (Primary Tests)
Uses subprocess to launch Cell-ACDC and verify:
- **Initialization**: Cell-ACDC can start through normal entry points
- **GUI Environment**: Qt libraries and GUI components initialize correctly  
- **Workflows**: Dataset loading and basic operations work as expected

This is the recommended approach and what CI/CD uses.

### `test_gui_automation.py` and `test_gui_advanced.py` (Legacy)
These files attempt direct GUI instantiation. They may have limited functionality due to circular import issues, but can be useful for:
- Direct Qt testing when the import issue is resolved
- Reference implementations for GUI interaction patterns

## Running Tests Locally

### Prerequisites

```bash
pip install pytest pytest-timeout pytest-qt pyautogui pillow scikit-image tifffile cellpose tables
pip install -e .
pip install -r requirements_gui_pyqt5.txt
```

### Generate Test Dataset

```bash
python tests/utils/generate_test_dataset.py
```

### Run Subprocess Tests (Recommended)

```bash
pytest tests/gui_automation/test_gui_subprocess.py -v
```

### Run All GUI Tests

```bash
# Subprocess tests (recommended)
pytest tests/gui_automation/test_gui_subprocess.py -v

# All tests (including legacy)
pytest tests/gui_automation/ -v --tb=short
```

### Run Specific Test

```bash
pytest tests/gui_automation/test_gui_subprocess.py::TestCellACDCInitialization::test_cellacdc_can_start_normally -v
```

## How Subprocess Tests Work

Each test spawns a new Python process that runs Cell-ACDC code:

```python
script = '''
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Your code here - runs with full Cell-ACDC initialization
from cellacdc import __version__
print(f"Cell-ACDC {__version__}")
'''

result = subprocess.run(
    [sys.executable, "-c", script],
    capture_output=True,
    timeout=30
)
```

This ensures:
- ✓ All `__main__.py` initialization code runs
- ✓ Patches for platform-specific issues (Windows DLLs, macOS symlinks, etc.) are applied
- ✓ No circular import issues
- ✓ Clean Python environment for each test

## GitHub Actions Workflow

The workflow `test_gui_automation.yml` runs on:
- Push to `main` and `optimizations` branches  
- Pull requests to these branches
- Manual trigger via `workflow_dispatch`

### CI Workflow Steps

1. Set up Python environment
2. Install dependencies (including optional ones like cellpose, tables)
3. Generate test dataset
4. Run subprocess tests
5. Upload artifacts on failure

### Environment Variables

- `QT_QPA_PLATFORM`: `offscreen` (headless rendering)
- `PYTEST_TIMEOUT`: 300 seconds

## Adding New Subprocess Tests

To add a new GUI test using the subprocess approach:

```python
def test_my_feature(self):
    """Test description."""
    logger.info("Testing my feature...")
    
    script = '''
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    # Your test code here
    from cellacdc import SomeModule
    result = SomeModule.do_something()
    if result == expected:
        print("SUCCESS")
    else:
        print("FAILED")
except Exception as e:
    print(f"FAILED: {e}")
'''
    
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert "SUCCESS" in result.stdout, f"Failed: {result.stderr}"
    logger.info("✓ Test passed")
```

## Advantages vs Direct Import

| Aspect | Subprocess | Direct Import |
|--------|-----------|---|
| Initialization | ✓ Full `__main__.py` setup | ✗ Skipped |
| DLL Patches | ✓ Applied automatically | ✗ Manual setup needed |
| Circular Imports | ✓ Avoided (separate process) | ✗ Can cause failures |
| State Isolation | ✓ Each test isolated | ✗ State can bleed between tests |
| Resource Cleanup | ✓ Automatic (process exit) | ✗ Requires manual cleanup |
| Speed | ~ 1-2 sec per test | ~ 0.5 sec per test |

## Troubleshooting

### ImportError: No module named 'cellacdc'

Ensure Cell-ACDC is installed in development mode:
```bash
pip install -e .
```

### Tests Skip or Timeout

Check that test dataset exists:
```bash
python tests/utils/generate_test_dataset.py
```

Increase timeout if needed:
```bash
pytest tests/gui_automation/ --timeout=600
```

### Circular Import in Legacy Tests

This is expected. The subprocess tests are the recommended approach.

## Future Enhancements

- [ ] GUI interaction testing (mouse clicks, keyboard input)
- [ ] Screenshot capture on failure
- [ ] Video recording of test sequences
- [ ] Dataset loading and segmentation workflow testing
- [ ] Export/import workflow testing
- [ ] Performance benchmarking
- [ ] Memory usage monitoring

## CI/CD Integration

The tests run automatically on:
- Every push to `main` or `optimizations`
- Every PR to these branches
- Manual workflow trigger

Artifacts (test data, screenshots) are uploaded for 7 days and available in GitHub Actions.

## Related Files

- [Workflow](.github/workflows/test_gui_automation.yml)
- [Test Dataset Generator](../utils/generate_test_dataset.py)
- [Main Entry Point](../../cellacdc/__main__.py)
- [Setup Functions](../../cellacdc/_run.py)
