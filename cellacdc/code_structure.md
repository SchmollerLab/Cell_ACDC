# Custom annotations in the main GUI

Currently we are implementing the following types of annotations:

1. Single time-point
2. Multiple time-points
3. Multiple values class

As of April 2022 we only implemented 1. Single time-point.

## Single time-point annotation

If the user loads time-lapse data then the annotation mode needs to be activated from the `modeComboBox` control. Otherwise it is always active.

### Construction

A custom annotation can be added by clicking on the `addCustomAnnotationAction`
tool button on the left side toolbar. This action is connected (with `triggered` slot) to the `addCustomAnnotation` function.

This function will open the dialog `apps.customAnnotationDialog` (with custom `exec_` function) and the window reference is stored in `addAnnotWin` attribute of the gui.

The `addAnnotWin` window has the following attributes created in its `closeEvent`:

- **symbol**: any of the `pyqtgraph` valid symbols (only the string inside the quotes, e.g., 'o' for circle, see `widgets.pgScatterSymbolsCombobox`)

- **keySequence**: a `QKeySequence` built with valid PyQt shortcut text, see [here](https://doc.qt.io/qt-5/qkeysequence.html#QKeySequence-1). Note that macOS shortcut strings are converted to valid PyQt string using the `widgets.macShortcutToWindows` function.

- **toolTip**: formatted text for `setToolTip` method of the button

- **state**: a dictionary with all the information needed to restore the annotations parameters, such as 'name', 'type', 'symbol'. Note that 'symbolColor' is a `QColor`.

With this info, when the window is closed with the `Ok` button, a tool button is added to the toolbar with the function `addCustomAnnotationButton`. This function adds a `widgets.customAnnotToolButton` (custom `paintEvent`).

The tool button reference is used as a key to create a dictionary inside the `customAnnotDict` (initialized in `__init__` method of the gui). This dictionary has four keys:

1. 'action' with the action linked to the tool button
2. 'state' with the dictionary of `addAnnotWin.state`
3. 'annotatedIDs' with a list of dictionaries, one dictionary per loaded position (same length as `self.data`). Each one of these dictionaries will be populated with the `frame_i` as key and, as value, the list of annotated IDs with that particular button.
4. 'scatterPlotItem', see below.

Next, the parameters of the annotation are saved as a json file to both `cellacdc/temp/custom_annotations.json` path (initialized as a global variable after the imports) and to the `<posData.images_path>/<basename>custom_annot_params.json` path (initialized in `load.loadData.buildPaths`). The saving is performed in the `saveCustomAnnot` function of the gui.

Finally, we create a `pg.ScatterPlotItem` and we add it to the `ax1` (left plot). We also add a column of 0s to the `acdc_df` filled with 0s and with column name = self.addAnnotWin.state['name'].

### Usage

The user clicks on the tool button of the annotation. This tool button is connected to `customAnnotButtonClicked`. Additionally, the tool button also has a right-click context menu with the following actions:

- **sigRemoveAction** --> `removeCustomAnnotButton`
- **sigKeepActiveAction** --> `customAnnotKeepActive`
- **sigModifyAction** --> `customAnnotModify`

At this point, the user can **RIGHT-click** on any segmented object, **only on the left image**. The click can be used also for undoing annotation.

The annotation is performed by the `doCustomAnnotation` function called in the case `elif isCustomAnnot:` of `gui_mousePressEventImg1`.

Note that `doCustomAnnotation` is also called at the end of `updateAllImages` to annotate every time the image changes.

The `doCustomAnnotation` function will check which button is active (from the keys of `customAnnotDict`). The checked button reference is used to access the `customAnnotDict[button]['annotatedIDs']` list of dictionaries. This list is indexed with `self.pos_i`. The resulting dictionary is indexed with `frame_i` to get the list of annotated IDs for the specific frame/position.

If the clicked ID is in `annotIDs_frame_i` then it is removed because the user is asking to undo the annotation, otherwise it is appended.

Next, we get the centroid coordinates of the annotated ID and we draw the scatter symbol. The 'scatterPlotItem' is stored in the `customAnnotDict` dictionary.

Finally, we reset to 0 the column in the acdc_df and we write 1 on the newly annotated IDs.

### Restoring of annotations after GUI is closed and re-opened

Steps for restoring:

1. `load.loadData` loads the attribute `posData.customAnnot` in the `loadOtherFiles` method. This attribute is set to the output of `load.read_json` function. The json file contains a dictionary where the keys are the names of each custom annotation, and the values are the `state` attribute of the `addAnnotWin` saved in `saveCustomAnnot` (with 'symbolColor' converted from QColor to rgb for json saving).
2. At the end of `load.loadData.loadOtherFiles` we call the function `load.loadData.getCustomAnnotatedIDs` that retrieves the annotated IDs from the acdc_df and it adds them to the `customAnnotIDs` dictionary (empty if acdc_df or customAnnot not found). The `customAnnotIDs` dictionary has the name of the saved custom annotations as keys, and, as values, a dictionary with frame_i as keys and the list of annotated IDs for that frame as values.
3. At the end of `gui.loadingDataCompleted` or in `gui.next_pos`/`gui.prev_pos` we call `addCustomAnnotationSavedPos` which iterates the items of `posData.customAnnot` where the keys are the names of the annotation and the values are the state saved in the json files:
    - For each annotation we re-build `symbolColor`, `keySequence` and `toolTip` and we add the tool button with `addCustomAnnotationButton`
    - If the tool button is not already present in the `customAnnotDict` we create it (see the **Construction** section above) and we add the `posData.customAnnotIDs` to the 'annotatedIDs' value `customAnnotDict`. Finaly, We add the scatter plot item with the function `addCustomAnnnotScatterPlot`
    - If the tool button is already present we only add `posData.customAnnotIDs` to the `customAnnotDict['annotatedIDs']` value.
