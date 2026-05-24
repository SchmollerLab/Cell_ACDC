"""Qt view adapter for lineage-tree interaction workflows."""

from __future__ import annotations

from qtpy.QtCore import Qt

from cellacdc import (
    disableWindow, exception_handler, html_utils, lineage_tree_cols, printl,
    widgets,
)
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    normal_division_lineage_tree,
)
from cellacdc.viewmodels.lineage_interactions_viewmodel import (
    LineageInteractionsViewModel,
)


class LineageInteractionsView:
    """Qt-facing adapter around lineage-tree interaction workflows."""

    LEGACY_METHODS = (
        'initLinTree',
        'propagateLinTreeAction',
        'resetLin_tree_future',
        'autoLinTree_df',
        'initMissingFramesLinTree',
        'viewLinTreeInfoAction',
        'askLineageTreeChanges',
        'repeat_click_and_backup',
        'getDistanceListMissingIDs',
        'find_mother_action',
        'annotate_unknown_lineage_action',
        'get_difference_table',
    )

    def __init__(self, host, view_model: LineageInteractionsViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    @exception_handler
    def initLinTree(self, force=False):
        """
        Initializes the lineage tree analysis.

        This method checks if the tracking has been previously checked and saved. If not, it displays a message to the user.
        It also prompts the user to go to the last annotated frame and restart the lineage tree analysis if necessary.
        Finally, it initializes the necessary data structures and updates the GUI.

        Returns
        -------
        proceed : bool
            True if the initialization is successful, nothing otherwise.
        """

        mode = str(self.modeComboBox.currentText())
        if not self.view_model.should_initialize(
            force=force,
            mode=mode,
            lineage_tree_exists=self.lineage_tree is not None,
        ):
            return

        posData = self.data[self.pos_i]
        last_tracked_i = self.get_last_tracked_i()
        defaultMode = self.view_model.default_mode_after_failed_init()
        if last_tracked_i == 0:
            # Display message to the user
            txt = html_utils.paragraph(
                'On this dataset either you <b>never checked</b> that the segmentation '
                'and tracking are <b>correct</b> or you did not save yet.<br><br>'
                'If you already visited some frames with "Segmentation and Tracking" '
                'mode save data before switching to "Normal division: Lineage Tree".<br><br>'
                'Otherwise you first have to check (and eventually correct) some frames '
                'in "Segmentation and Tracking" mode before proceeding '
                'with lineage tree analysis.')
            msg = widgets.myMessageBox()
            msg.critical(
                self.host, 'Tracking was never checked', txt
            )
            self.modeComboBox.setCurrentText(defaultMode)
            return

        proceed = True
        last_lin_tree_frame_i = self.view_model.last_annotated_frame_index(
            posData.allData_li
        )

        if last_lin_tree_frame_i == 0:
            # Remove undoable actions from segmentation mode
            posData.UndoRedoStates[0] = []
            self.undoAction.setEnabled(False)
            self.redoAction.setEnabled(False)

        if posData.frame_i > last_lin_tree_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_lin_tree_frame_i+1}.<br><br>
                Do you want to restart lineage tree analysis from frame
                {last_lin_tree_frame_i+1}?<br>
            """)
            _, yesButton, stayButton = msg.warning(
                self.host, 'Go to last annotated frame?', txt,
                buttonsTexts=(
                    'Cancel', f'Yes, go to frame {last_lin_tree_frame_i+1}',
                    'No, stay on current frame')
            )
            if yesButton == msg.clickedButton:
                msg = 'Looking good!'
                self.last_lin_tree_frame_i = last_lin_tree_frame_i
                posData.frame_i = last_lin_tree_frame_i
                self.titleLabel.setText(msg, color=self.titleColor)
                self.get_data(lin_tree_init=False)
                self.updateAllImages() # i dont think I need to change this
                self.updateScrollbars() # i dont think I need to change this
            elif stayButton == msg.clickedButton:
                self.initMissingFramesLinTree(posData.frame_i) #!!!
                last_lin_tree_frame_i = posData.frame_i
                msg = 'Lineage tree analysis initialised!'
                self.titleLabel.setText(msg, color='g')
            elif msg.cancel:
                msg = 'Lineage tree analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(defaultMode)
                proceed = False
                return

        elif posData.frame_i < last_lin_tree_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_lin_tree_frame_i+1}.<br><br>
                Do you want to restart lineage tree analysis from frame
                {last_lin_tree_frame_i+1}?<br>
            """)
            goTo_last_annotated_frame_i = msg.question(
                self.host, 'Go to last annotated frame?', txt,
                buttonsTexts=('Yes', 'No', 'Cancel')
            )[0]
            if goTo_last_annotated_frame_i == msg.clickedButton:
                msg = 'Looking good!'
                self.titleLabel.setText(msg, color=self.titleColor)
                self.last_lin_tree_frame_i = last_lin_tree_frame_i
                posData.frame_i = last_lin_tree_frame_i
                self.get_data(lin_tree_init=False)
                self.updateAllImages() # i dont think I need to change this
                self.updateScrollbars() # i dont think I need to change this
            elif msg.cancel:
                msg = 'Lineage tree analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(defaultMode)
                proceed = False
                return
        else:
            self.get_data(lin_tree_init=False)

        self.last_lin_tree_frame_i = last_lin_tree_frame_i

        self.navigateScrollBar.setMaximum(last_lin_tree_frame_i+1)
        self.navSpinBox.setMaximum(last_lin_tree_frame_i+1)

        if self.lineage_tree is None or force:
            self.store_data(autosave=False)
            self.get_data(lin_tree_init=False)
            self.lineage_tree = normal_division_lineage_tree(gui=self.host)

            msg = 'Lineage tree analysis initialized!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)

        return proceed

    @disableWindow
    def propagateLinTreeAction(self, dummy_for_button=None):
        """
        Propagates the lineage tree based on the current frame_i. Used in self.propagateLinTreeButton.
        """
        posData = self.data[self.pos_i]
        self.lineage_tree.propagate(posData.frame_i)
        if posData.frame_i == self.original_df_lin_tree_i:
            self.original_df_lin_tree = posData.allData_li[posData.frame_i]['acdc_df'].copy()

        self.logger.info('Lineage tree propagated.')

    def resetLin_tree_future(self):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        result = (
            self.host.view_model.lineage.remove_future_lineage_tree_annotations(
                posData.allData_li,
                lineage_tree_cols,
                frame_i,
                size_t=posData.SizeT,
            )
        )

        if self.lineage_tree is not None:
            self.lineage_tree.frames_for_dfs.discard(frame_i)
        for i, acdc_df in result.acdc_dfs_by_frame.items():
            posData.allData_li[i]['acdc_df'] = acdc_df

    def autoLinTree_df(self, enforceAll=False):
        """Automatically generates a lineage tree dataframe.

        This method generates a lineage tree dataframe based on the current mode and data.
        It checks if the mode is set to 'Normal division: Lineage tree' and if the current frame
        is not already processed. If the conditions are met, it retrieves the necessary data
        from the current position data and previous position data, and passes it to the
        `real_time` method of the `lineage_tree` object. Finally, it converts the lineage tree
        to an ACDC dataframe and adds the current frame to the set of frames that have been
        processed.

        Parameters
        ----------
        enforceAll : bool, optional
            If True, enforces processing of all frames, even if they have been processed before.
            If False, only processes frames that have not been processed before. Default is False.

        Returns
        -------
        bool
            True if there are not enough G1 cells for lineage tree generation, False otherwise.
        bool
            True if the lineage tree generation should proceed, False otherwise.
        """
        proceed = True
        notEnoughG1Cells = False
        mode = str(self.modeComboBox.currentText())

        # Skip if not the right mode
        if not self.view_model.should_process_auto_frame(
            mode=mode,
            frame_i=self.data[self.pos_i].frame_i,
            processed_frames=self.lineage_tree.frames_for_dfs,
        ):
            return notEnoughG1Cells, proceed

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i

        # Make sure that this is a visited frame in segmentation tracking mode
        if posData.allData_li[frame_i]['labels'] is None: # may need to change this
            proceed = self.warnFrameNeverVisitedSegmMode()
            return notEnoughG1Cells, proceed

        self.store_data(autosave=False)
        self.get_data()
        lab = posData.lab
        prev_lab = posData.allData_li[frame_i-1]['labels']
        rp = posData.rp
        prev_rp = posData.allData_li[frame_i-1]['regionprops']

        self.lineage_tree.real_time(frame_i, lab, prev_lab, rp=rp, prev_rp=prev_rp)
        self.store_data()

    def initMissingFramesLinTree(self, current_frame_i): # done Need to add partially missing previous frames and loading
        """
        When not starting from the first frame, automatically creates lineage tree dfs for all "skipped" frames and initializes the tree if not done so before.

        Parameters
        ----------
        current_frame_i : int
            The index of the current frame.

        Returns
        -------
        None

        Notes
        -----
        This method initializes the lineage tree annotations of missing past frames. If the lineage tree has not been initialized before, it creates a new lineage tree based on the labels of the first frame. It then iterates over the missing frames and updates the lineage tree with the labels and region properties of each frame.
        """

        self.logger.info(
            'Initialising lineage tree annotations of missing past frames...'
        )

        self.store_data(autosave=False)
        self.get_data()

        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i

        if not self.lineage_tree: # init lin tree if not done already
            self.lineage_tree = normal_division_lineage_tree(
                gui=self.host
            ) # here frame_i!=0

        present_frames = list(self.lineage_tree.frames_for_dfs) if self.lineage_tree else []
        present_frames = [] if not present_frames else present_frames # deal with None
        missing_frames = self.view_model.missing_frame_indices(
            current_frame_i,
            present_frames,
        )

        for frame_i in missing_frames:
            lab = posData.allData_li[frame_i]['labels']
            prev_lab = posData.allData_li[frame_i-1]['labels']
            rp = posData.allData_li[frame_i]['regionprops']
            prev_rp = posData.allData_li[frame_i-1]['regionprops']
            # i might need to change this if I need support for only partially missing frames... Although I probably never have to care about that though
            self.lineage_tree.real_time(frame_i, lab, prev_lab, rp=rp, prev_rp=prev_rp)

        posData.frame_i = current_frame_i
        self.store_data()

    def viewLinTreeInfoAction(self):
        mode = str(self.modeComboBox.currentText())
        if not self.view_model.is_lineage_mode(mode):
            self.logger.info('This action is only available in the "Normal division: Lineage tree" mode.')
            return

        if not self.lineage_tree:
            self.logger.info('No lineage tree found.')
            return

        posData = self.data[self.pos_i]

        if self.original_df_lin_tree_i != posData.frame_i:
            # could be that this is not entirley true and self.curr_original_df_i just didnt get set right though!
            txt_changes = '<br>No changes were made in this frame.<br><br>'

        else:
            result = self.get_difference_table(return_css_separated=True)

            if result is None:
                txt_changes = 'No changes were made in this frame.'
            else:
                css, txt_changes = result

        txt_changes = '<b>Changes made in this frame</b>:' + txt_changes + '<br><br>'

        cells_with_parent, orphan_cells, lost_cells = self.lineage_tree.export_lin_tree_info(posData.frame_i)

        if orphan_cells == []:
            txt_orphan_cells = 'No orphan Cells!'
        else:
            txt_orphan_cells = ', '.join([str(cell) for cell in orphan_cells])
        txt_orphan = f'<b>Orphan cells</b>:<br>{txt_orphan_cells}<br><br>'

        lost_cells = list(lost_cells)
        if lost_cells == []:
            txt_lost_cells = 'No lost Cells!'
        else:
            txt_lost_cells = ', '.join([str(cell) for cell in lost_cells])
        txt_lost = f'<b>Lost cells</b>:<br>{txt_lost_cells}<br><br>'

        if cells_with_parent == []:
            table_cells_with_parent = '<br>No cells with parents!'
        else:
            table_cells_with_parent = """<table>
                        <tr>
                            <th>Parent ID</th>
                            <th>ID</th>
                        </tr>"""

            for cell, parent in cells_with_parent:
                table_cells_with_parent += f'''<tr>
                                <td>{parent}</td>
                                <td>{cell}</td>
                            </tr>'''
            table_cells_with_parent += '</table>'

        txt_cells_with_parents = f'<b>Cells with parents</b>:{table_cells_with_parent} <br><br>'

        css = r'''
                <style>
                    table, th, td {
                        border: 1px solid grey;
                        border-collapse: collapse;
                    }
                    th, td {
                        padding: 5px;
                    }
                </style>
            '''

        txt = css + html_utils.paragraph(txt_changes + txt_orphan + txt_lost + txt_cells_with_parents)

        msg = widgets.myMessageBox()
        msg.information(self.host,
                'lineage tree information',
                txt
                )

    @disableWindow
    def askLineageTreeChanges(self):
        """
        Asks the user for changes in the lineage tree.

        This method is called when the user selects the 'Normal division: Lineage tree' mode.
        It compared the backed up df (self.original_df from repeat_click_and_backup) with the current df (self.lineage_tree.export_df(posData.frame_i)) and propts the user to keep, propagate or discard the changes.

        """
        mode = str(self.modeComboBox.currentText())
        if not self.view_model.is_lineage_mode(mode):
            return

        if not self.lineage_tree:
            return

        posData = self.data[self.pos_i]

        if self.original_df_lin_tree_i is not None and self.original_df_lin_tree_i != posData.frame_i:
            printl("!This should not happen!")
            self.store_data(autosave=False)
            og_frame = posData.frame_i
            posData.frame_i = self.original_df_lin_tree_i
            self.get_data()
            self.logger.info('Lineage tree changes were not propagated, going back to original frame.')
            self.askLineageTreeChanges()
            self.store_data(autosave=False)
            posData.frame_i = og_frame
            self.get_data()
            return

        result = self.get_difference_table(return_css_separated=True, return_differece=True)
        if result is None:
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            return

        css, txt, differences = result
        changed_IDs = differences['Cell_ID'].unique()

        if posData.frame_i == max(self.lineage_tree.frames_for_dfs):
            # here we can just propagate the cahnged. This is super fast, since there is no recursion, no children and fast finding of parents
            self.lineage_tree.propagate(posData.frame_i, relevant_cells=changed_IDs)
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            return

        txt = txt + 'Do you want to keep, propgagte or discard the changes?'
        txt = css + html_utils.paragraph('<b>Changes made in this frame</b><br>' + txt)

        msg = widgets.myMessageBox()

        propagate_btn, discard_btn, _ = msg.question(self.host,
                      'Changes in lineage tree',
                      txt,
                      buttonsTexts=('Propagate', 'Discard', 'Cancel'),)

        if msg.clickedButton == propagate_btn:
            self.lineage_tree.propagate(posData.frame_i, relevant_cells=changed_IDs)
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            self.logger.info('Lineage tree propagated.')

        elif msg.clickedButton == discard_btn:
            posData.allData_li[posData.frame_i]['acdc_df'] = self.original_df_lin_tree.copy()
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            self.logger.info('Lineage tree changes discarded.')


        elif msg.cancel:
            # Go back to current frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph('''
            Changes were kept but not propagated!
            Please make sure to come back and propagate them,
            otherwise your table might be inconsistent!
            There is a button for this next to the edit buttons.
            <b>Please also do not visit new frames!</b>

            ''')
            msg.warning(self.host, 'Changes kept but not propagated!', txt)
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            self.logger.info('Lineage tree changes discarded.')

    def repeat_click_and_backup(self, posData, event, ydata, xdata):
        """
        This function is part of the lin_tree edit functionality.
        It handles the back up of the original self.lineage_tree.lineage_list
        df and the repeated clicking on the same ID to cycle through pssible mothers.

        Parameters
        ----------
        posData : cellacdc.load.loadData
            The position data.
        event : QtGui.QMouseEvent
            The event object.
        ydata : int
            The y-coordinate data.
        xdata : int
            The x-coordinate data.

        Returns
        -------
        tuple
            A tuple containing the point(tuple: (x, y) coords) and ID of clicked cell.
        """
        should_log_reset = (
            self.original_df_lin_tree is not None
            and self.original_df_lin_tree_i != posData.frame_i
        )
        if self.view_model.should_backup_original(
            self.original_df_lin_tree_i,
            posData.frame_i,
        ):
            if should_log_reset:
                self.logger.info(
                    '[WARNING]: !!! Original lineage tree df changed, '
                    'resetting original_df_lin_tree !!!'
                )
            self.original_df_lin_tree = posData.allData_li[posData.frame_i]['acdc_df'].copy()
            self.original_df_lin_tree_i = posData.frame_i

        if not self.right_click_ID:
            self.right_click_i = 0
            self.right_click_ID = 0

        x, y = event.pos().x(), event.pos().y()
        point = int(x), int(y)
        ID = self.get_2Dlab(posData.lab)[ydata, xdata]

        if ID == 0:
            return None, None

        if self.right_click_ID != ID:
            self.right_click_i = 0
            self.right_click_ID = ID
            self.original_mother_skipped = False
        elif event.modifiers() & Qt.ShiftModifier:
            self.right_click_i -= 1
        else:
            self.right_click_i += 1

        return point, ID

    def getDistanceListMissingIDs(self, point, ID):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        if self.getDistanceListMissingIDsCachedFrame != frame_i:
            self.distanceListMissingIDs = dict()
            self.getDistanceListMissingIDsCachedFrame = frame_i
            # self.store_data(autosave=False)
            # self.get_data()

        if ID not in self.distanceListMissingIDs.keys():
            prev_rp = posData.allData_li[frame_i-1]['regionprops']
            relevant_rp = [
                obj for obj in prev_rp if obj.label not in posData.IDs
            ]
            len_relevant_rp = len(relevant_rp)
            if len_relevant_rp == 0:
                self.logger.info('No missing IDs found in previous frame.')
                return []
            elif len_relevant_rp == 1:
                self.distanceListMissingIDs[ID] = [relevant_rp[0].label]
                return [relevant_rp[0].label]
            else:
                sorted_missing_IDs = self.host.view_model.lineage.sort_ids_by_distance(
                    relevant_rp, point=point
                )
                self.distanceListMissingIDs[ID] = sorted_missing_IDs
                return sorted_missing_IDs
        else:
            return self.distanceListMissingIDs[ID]

    def find_mother_action(self, posData, event, ydata, xdata):
        """
        This function is part of the lin_tree edit functionality.
        Associated with the right-click action of the 'findNextMotherButton' button.
        Handles the right click action, which cycles through possible mothers of the clicked cell.
        Changes the parent ID of the clicked cell to the next possible mother in self.lineage_tree.lineage_list.

        Parameters
        ----------
        posData : cellacdc.load.loadData
            The position data object.
        event : QtGui.QMouseEvent
            The event object.
        ydata : int
            The y-coordinate data.
        xdata : int
            The x-coordinate data.
        """
        point, ID = self.repeat_click_and_backup(posData, event, ydata, xdata)

        if point is None:
            return
        posData = self.data[self.pos_i]
        acdc_df_frame = posData.allData_li[posData.frame_i]['acdc_df']
        filtered_IDs = self.getDistanceListMissingIDs(point, ID)
        if len(filtered_IDs) == 0:
            self.logger.info('No mother candidates found.')
            return

        i = self.view_model.next_candidate_index(
            self.right_click_i,
            len(filtered_IDs),
        )
        new_mother = filtered_IDs[i]

        if self.view_model.should_skip_original_mother(
            acdc_df_frame.loc[ID]['parent_ID_tree'],
            new_mother,
            original_mother_skipped=self.original_mother_skipped,
        ): # if a mother is already present, skip it
            self.right_click_i += 1
            self.original_mother_skipped = True

            i = self.view_model.next_candidate_index(
                self.right_click_i,
                len(filtered_IDs),
            )
            new_mother = filtered_IDs[i]

        acdc_df_frame.at[ID, 'parent_ID_tree'] = new_mother # update mother in the df, no need to propagate or stuff lile this
        # dont need to update alldata_li as acdc_df_frame is just a view
        self.drawAllLineageTreeLines()

    def annotate_unknown_lineage_action(self, posData, event, ydata, xdata):
        """
        This function is part of the lin_tree edit functionality.
        Associated with the right-click action of the 'unknownLineageButton' button.
        Annotates an unknown lineage by setting its parent ID to -1 in the lineage tree (self.lineage_tree.lineage_list)

        Parameters
        ----------
        posData : cellacdc.load.loadData
            The position data.
        event : QtGui.QMouseEvent
            The event that triggered the annotation.
        ydata : int
            The y-coordinate data.
        xdata : int
            The x-coordinate data.
        """
        point, ID = self.repeat_click_and_backup(posData, event, ydata, xdata)

        if point is None:
            return
        posData = self.data[self.pos_i]
        acdc_df_frame = posData.allData_li[posData.frame_i]['acdc_df']
        acdc_df_frame.at[ID, 'parent_ID_tree'] = -1
        self.drawAllLineageTreeLines()

    @disableWindow
    def get_difference_table(self, return_css_separated=False, return_differece=False):

        if self.original_df_lin_tree is None:
            return

        posData = self.data[self.pos_i]

        new_df = posData.allData_li[posData.frame_i]['acdc_df']
        original_df = self.original_df_lin_tree.copy()

        if original_df.equals(new_df):
            return

        differences = self.view_model.parent_id_differences(
            original_df,
            new_df,
            self.host.view_model.tables.checked_reset_index_cell_id,
        )
        if differences is None:
            return

        txt = """<table>
                    <tr>
                        <th>ID</th>
                        <th>old parent --></th>
                        <th>new parent</th>
                    </tr>"""

        for diff in differences.itertuples():
            ID = str(int(diff.Cell_ID))
            old_parent = str(int(diff.self))
            new_parent = str(int(diff.other))

            txt += f'''<tr>
                            <td>{ID}</td>
                            <td>{old_parent}</td>
                            <td>{new_parent}</td>
                        </tr>'''
        txt += '</table>'

        css = r'''
            <style>
                table, th, td {
                    border: 1px solid grey;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 5px;
                }
            </style>
        '''
        if return_css_separated and not return_differece:
            return css, txt
        elif return_css_separated and return_differece:
            return css, txt, differences
        elif not return_css_separated and return_differece:
            return txt, differences
        else:
            txt = css + html_utils.paragraph(txt)
            return txt
