"""Lineage tree mode of the main GUI. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
from qtpy.QtCore import Qt
from cellacdc import exception_handler, disableWindow
from cellacdc import html_utils
from cellacdc.myutils import table_utils
from .. import helpers as gui_helpers
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    normal_division_lineage_tree, has_lineage_tree_values,
    lineage_tree_columns, lineage_tree_annotated_frames,
    LINEAGE_TREE_SHARED_CCA_COLS
)


class LineageTreeGuiElements:
    """Lineage tree mode: find mother and unknown lineage clicks, backup
    and change detection, automatic lineage tree and reset of future
    lineage annotations.
    """

    def repeat_click_and_backup(self, posData, event, ydata, xdata):
        """
        This function is part of the lin_tree edit functionality. 
        It handles the back up of the original self.lineage_tree.lineage_list 
        df and the repeated clicking on the same ID to cycle through pssible mothers.

        A backup of another frame holds edits not checked yet (the frames
        scroll bar and spin box change frame without checking them): they
        are checked on their frame first (`askLineageTreeChanges`, which
        reloads the current frame afterwards), then the current frame is
        backed up.

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
        if (
                self.original_df_lin_tree is not None
                and self.original_df_lin_tree_i != posData.frame_i
            ):
            # Edits of another frame: check them before replacing the backup
            # (the prompt may propagate them, also into this frame)
            self.askLineageTreeChanges()
            self.updateAllImages()

        if (
                self.original_df_lin_tree is None
                or self.original_df_lin_tree_i != posData.frame_i
            ):
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

        if ID not in self.distanceListMissingIDs.keys():
            prev_rp = posData.allData_li[frame_i-1]['regionprops']
            relevant_IDs = [
                ID for ID in prev_rp.IDs if ID not in posData.IDs
            ]
            len_relevant_IDs = len(relevant_IDs)
            if len_relevant_IDs == 0:
                self.logger.info('No missing IDs found in previous frame.')
                return []
            elif len_relevant_IDs == 1:
                self.distanceListMissingIDs[ID] = relevant_IDs
                return relevant_IDs
            else:
                sorted_missing_IDs = gui_helpers.sort_IDs_dist(prev_rp, point=point, relevant_IDs=relevant_IDs)
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

        if 'parent_ID_tree' not in acdc_df_frame.columns:
            self.logger.info('No parent_ID_tree column found in the current frame.')
            return

        # check that the clicked cell is a new ID
        if ID not in posData.new_IDs:
            self.logger.info('Clicked cell is not a new ID.')
            return

        filtered_IDs = self.getDistanceListMissingIDs(point, ID)
        if len(filtered_IDs) == 0:
            self.logger.info('No mother candidates found.')
            return

        i = self.right_click_i % len(filtered_IDs)
        i = abs(i)  # Ensure i is non-negative
        new_mother = filtered_IDs[i]

        if (acdc_df_frame.loc[ID]['parent_ID_tree'] == new_mother 
            and self.original_mother_skipped == False): # if a mother is already present, skip it 
            self.right_click_i += 1
            self.original_mother_skipped = True

            i = self.right_click_i % len(filtered_IDs)
            i = abs(i)  # Ensure i is non-negative
            new_mother = filtered_IDs[i]

        acdc_df_frame.at[ID, 'parent_ID_tree'] = new_mother # update mother in the df, no need to propagate or stuff lile this
        
        acdc_df_frame_prev = posData.allData_li[posData.frame_i - 1]['acdc_df']
        acdc_df_frame.at[ID, 'generation_num_tree'] = (
            acdc_df_frame_prev.loc[new_mother]['generation_num_tree'] + 1
            ) # update generation number in the df
        # dont need to update alldata_li as acdc_df_frame is just a view
        self.drawAllLineageTreeLines()
        self.setAllTextAnnotations()

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
        
        if ID not in posData.new_IDs:
            self.logger.info('Clicked cell is not a new ID.')
            return
            
        posData = self.data[self.pos_i]
        acdc_df_frame = posData.allData_li[posData.frame_i]['acdc_df']
        acdc_df_frame.at[ID, 'parent_ID_tree'] = -1
        acdc_df_frame.at[ID, 'generation_num_tree'] = 1
        self.drawAllLineageTreeLines()
        self.setAllTextAnnotations()

    def askInitLinTreeFirstFrame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != 'Normal division: Lineage tree':
            return True
        
        posData = self.data[self.pos_i]
        if posData.frame_i != 0:
            return True
        
        if self.lineage_tree is None:
            self.initLinTree() 
        
        return True

    def clearLineageTreeBackup(self):
        """Forget the frame table backed up by `repeat_click_and_backup`
        (first right-click of a lineage tree edit), which
        `askLineageTreeChanges` compares with the current table.

        Call it whenever `self.lineage_tree` is set to None and when new data
        is loaded. A backup left from a lineage tree session that ended would
        be checked on the next frame change or save in lineage tree mode,
        when the backed up frame may have lost its annotations (reset after
        a segmentation edit) or belong to other data.
        """
        self.original_df_lin_tree = None
        self.original_df_lin_tree_i = None

    def lineageTreeBackupMismatch(self, posData):
        """Why the table backed up by `repeat_click_and_backup` cannot be
        compared with the current table of its frame anymore.

        The comparison (`get_difference_table`) needs the `parent_ID_tree`
        values of the same objects in both tables. The backup is stale when
        its frame does not exist or was not loaded, when the backup or the
        frame has no lineage tree values (e.g. the frame's annotations were
        reset after a segmentation edit) or when the frame's objects changed.

        Parameters
        ----------
        posData : cellacdc.load.loadData
            The position data of the backup.

        Returns
        -------
        str or None
            The reason, or None when the backup can be compared (or there
            is no backup).
        """
        from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
            has_lineage_tree_values
        )
        frame_i = self.original_df_lin_tree_i
        backup_df = self.original_df_lin_tree
        if frame_i is None or backup_df is None:
            return

        if frame_i >= len(posData.allData_li):
            return f'frame {frame_i+1} does not exist'

        frame_data = posData.allData_li[frame_i]
        df = frame_data['acdc_df']
        if frame_data['labels'] is None or df is None:
            return f'frame {frame_i+1} is not loaded'

        for table in (backup_df, df):
            if (
                'parent_ID_tree' not in table.columns
                or not has_lineage_tree_values(table)
            ):
                return f'frame {frame_i+1} has no lineage tree annotations'

        backup_IDs = table_utils.checked_reset_index_Cell_ID(backup_df).index
        IDs = table_utils.checked_reset_index_Cell_ID(df).index
        if set(backup_IDs) != set(IDs):
            return f'the objects of frame {frame_i+1} changed'

    def clearStaleLineageTreeBackup(self, posData):
        """Clear the backup of `repeat_click_and_backup` if it cannot be
        compared with the current table of its frame anymore (see
        `lineageTreeBackupMismatch`), so that its check neither raises nor
        brings back the annotations of a reset frame.

        Parameters
        ----------
        posData : cellacdc.load.loadData
            The position data of the backup.

        Returns
        -------
        bool
            True if the backup was stale and was cleared.
        """
        reason = self.lineageTreeBackupMismatch(posData)
        if reason is None:
            return False

        self.logger.info(
            '[WARNING]: Lineage tree edits of frame '
            f'{self.original_df_lin_tree_i+1} are not checked for changes '
            f'to propagate ({reason}).'
        )
        self.clearLineageTreeBackup()
        return True

    @disableWindow
    def get_difference_table(self, return_css_separated=False, return_differece=False):

        if self.original_df_lin_tree is None:
            return

        posData = self.data[self.pos_i]
        if self.clearStaleLineageTreeBackup(posData):
            return
        
        new_df = posData.allData_li[posData.frame_i]['acdc_df']
        original_df = self.original_df_lin_tree.copy()

        if new_df is None or original_df.equals(new_df):
            return

        compare_columns = ['parent_ID_tree']
        for df in (original_df, new_df):
            if any(col not in df.columns for col in compare_columns):
                # No lineage tree annotations to compare, e.g. removed by
                # `resetLin_tree_future` after a segmentation edit
                return

        new_df = table_utils.checked_reset_index_Cell_ID(new_df)
        original_df = table_utils.checked_reset_index_Cell_ID(original_df)
        # Segmentation edits can add or remove IDs after the backup:
        # compare the cells present in both tables
        IDs = original_df.index.intersection(new_df.index)
        new_df = new_df.loc[IDs, compare_columns].sort_index()
        original_df = original_df.loc[IDs, compare_columns].sort_index()

        differences = original_df.compare(new_df)
        if differences.empty:
            return
        
        differences = table_utils.checked_reset_index_Cell_ID(differences)
 
        differences = differences['parent_ID_tree']
        differences = differences.reset_index()

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

    def viewLinTreeInfoAction(self):
        mode = str(self.modeComboBox.currentText())
        if mode != 'Normal division: Lineage tree':
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

        msg = widgets_message_box.myMessageBox()
        msg.information(self,
                'lineage tree information', 
                txt
                )

    @disableWindow
    def askLineageTreeChanges(self, isLeavingMode=False):
        """
        Asks the user for changes in the lineage tree.

        This method is called in the 'Normal division: Lineage tree' mode
        before changing frame (next/previous frame) and saving, by
        `changeMode` when leaving that mode and by `repeat_click_and_backup`
        before it backs up another frame.
        It compares the backed up df (self.original_df_lin_tree from
        repeat_click_and_backup) with the current df of that frame and
        prompts the user to propagate or discard the changes, or to keep
        them without propagating (Cancel). A backup of another frame than
        the current one (frame changed with the scroll bar or spin box) is
        checked on its frame, then the current frame is loaded again. The
        backup is cleared afterwards, also when it no longer matches the
        frame (see `lineageTreeBackupMismatch`).

        Parameters
        ----------
        isLeavingMode : bool, optional
            True when `changeMode` leaves the lineage tree mode: the mode
            combo box already shows the new mode, so the mode is not
            checked. Cancel cannot cancel the mode switch there (the combo
            box already changed): the changes are kept without propagating
            them, as before a frame change. Default is False.
        """
        mode = str(self.modeComboBox.currentText())
        if mode != 'Normal division: Lineage tree' and not isLeavingMode:
            return
        
        if not self.lineage_tree:
            self.clearLineageTreeBackup()
            return
        
        posData = self.data[self.pos_i]
        if self.clearStaleLineageTreeBackup(posData):
            return
        
        if self.original_df_lin_tree_i is not None and self.original_df_lin_tree_i != posData.frame_i:
            # Edits of another frame: the frames scroll bar and spin box
            # change frame without checking them
            self.store_data(autosave=False)
            og_frame = posData.frame_i
            posData.frame_i = self.original_df_lin_tree_i
            try:
                self.get_data()
                self.logger.info(
                    'Lineage tree changes were not propagated, '
                    'going back to original frame.'
                )
                self.askLineageTreeChanges(isLeavingMode=isLeavingMode)
                self.store_data(autosave=False)
            except Exception as err:
                # Do not fail again on every frame change and save
                self.clearLineageTreeBackup()
                raise err
            finally:
                posData.frame_i = og_frame
                self.get_data()
            return

        try:
            result = self.get_difference_table(
                return_css_separated=True, return_differece=True
            )
        except Exception as err:
            # Do not fail again on every frame change and save
            self.clearLineageTreeBackup()
            raise err

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

        msg = widgets_message_box.myMessageBox()

        propagate_btn, discard_btn, _ = msg.question(self,
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
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph('''
            Changes were kept but not propagated!
            Please make sure to come back and propagate them,
            otherwise your table might be inconsistent!
            There is a button for this next to the edit buttons.
            <b>Please also do not visit new frames!</b>
            
            ''')
            msg.warning(self, 'Changes kept but not propagated!', txt)
            self.original_df_lin_tree = None
            self.original_df_lin_tree_i = None
            self.logger.info('Lineage tree changes kept but not propagated.')

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
        # The tree does not exist yet: without `lin_tree_init=False`
        # get_data would call `initLinTree` again, repeating its prompt
        self.get_data(lin_tree_init=False)

        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i

        if not self.lineage_tree: # init lin tree if not done already
            self.lineage_tree = normal_division_lineage_tree(gui=self) # here frame_i!=0

        missing_frames = list(range(current_frame_i+1))
        present_frames = list(self.lineage_tree.frames_for_dfs) if self.lineage_tree else []
        present_frames = [] if not present_frames else present_frames # deal with None
        missing_frames = [frame_i for frame_i in missing_frames if frame_i not in present_frames]
        missing_frames.sort()

        for frame_i in missing_frames:
            lab = posData.allData_li[frame_i]['labels']
            prev_lab = posData.allData_li[frame_i-1]['labels']
            rp = posData.allData_li[frame_i]['regionprops']
            prev_rp = posData.allData_li[frame_i-1]['regionprops']
            # i might need to change this if I need support for only partially missing frames... Although I probably never have to care about that though
            self.lineage_tree.real_time(frame_i, lab, prev_lab, rp=rp, prev_rp=prev_rp)

        posData.frame_i = current_frame_i
        self.store_data()

    def autoLinTree_df(self, enforceAll=False):
        """Automatically generates a lineage tree dataframe.

        This method generates a lineage tree dataframe based on the current mode and data.
        It checks if the mode is set to 'Normal division: Lineage tree' and if the current frame
        is not already processed. If the conditions are met, it retrieves the necessary data
        from the current position data and previous position data, and passes it to the
        `real_time` method of the `lineage_tree` object. Finally, it converts the lineage tree
        to an ACDC dataframe and adds the current frame to the set of frames that have been
        processed.

        A frame is never annotated from an unannotated previous frame (every
        cell would become a new root): the frames missing between the last
        annotated frame and the current one are annotated first, in order
        (`annotateLinTreeFramesBefore`).

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
        if mode != 'Normal division: Lineage tree':
            return notEnoughG1Cells, proceed
        
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i

        if frame_i in self.lineage_tree.frames_for_dfs:
            return notEnoughG1Cells, proceed

        # Make sure that this is a visited frame in segmentation tracking mode
        if posData.allData_li[frame_i]['labels'] is None: # may need to change this
            proceed = self.warnFrameNeverVisitedSegmMode()
            return notEnoughG1Cells, proceed
        
        self.store_data(autosave=False)
        self.get_data()
        if not self.annotateLinTreeFramesBefore(frame_i):
            return notEnoughG1Cells, proceed

        lab = posData.lab
        prev_lab = posData.allData_li[frame_i-1]['labels']
        rp = posData.rp
        prev_rp = posData.allData_li[frame_i-1]['regionprops']

        self.lineage_tree.real_time(frame_i, lab, prev_lab, rp=rp, prev_rp=prev_rp)
        self.store_data()
        return notEnoughG1Cells, proceed

    def annotateLinTreeFramesBefore(self, frame_i):
        """Annotate the lineage tree of the frames missing between the last
        annotated frame before `frame_i` and `frame_i` (excluded), in order,
        so that `frame_i` is annotated from an annotated previous frame.

        A lineage tree reset with a live tree (`resetLin_tree_future`, e.g.
        after a segmentation edit in lineage tree mode) leaves the current
        frame unannotated: its labels can still change after the warning
        (the post-processing dialog warns before it sets the new masks), so
        it is annotated here, from its stored labels, when the next frame
        is entered (`autoLinTree_df`).

        Parameters
        ----------
        frame_i : int
            The frame about to be annotated.

        Returns
        -------
        bool
            False if a missing frame was never visited (no stored labels or
            table): `frame_i` must not be annotated then.
        """
        posData = self.data[self.pos_i]
        last_annotated_i = max(
            (i for i in self.lineage_tree.frames_for_dfs if i < frame_i),
            default=0
        )
        for i in range(last_annotated_i+1, frame_i):
            data_dict = posData.allData_li[i]
            if data_dict['labels'] is None or data_dict['acdc_df'] is None:
                self.logger.info(
                    f'[WARNING]: Frame n. {i+1} was never visited: the '
                    f'lineage tree of frames n. {i+1}-{frame_i+1} cannot be '
                    'annotated.'
                )
                return False

            self.logger.info(
                f'Annotating lineage tree of frame n. {i+1} again...'
            )
            prev_data_dict = posData.allData_li[i-1]
            self.lineage_tree.real_time(
                i, data_dict['labels'], prev_data_dict['labels'],
                rp=data_dict['regionprops'],
                prev_rp=prev_data_dict['regionprops']
            )
        return True

    def get_last_lin_tree_frame_i(self):
        """Index of the last frame with lineage tree annotations, -1 if no
        frame has them.

        Same annotated frames as `initLinTree` and the lineage tree's
        `frames_for_dfs` (`lineage_tree_annotated_frames`: contiguous from
        frame 0, frames without objects do not end them), but -1 when none
        of them has lineage tree values (e.g. only empty first frames).

        Returns
        -------
        int
        """
        posData = self.data[self.pos_i]
        acdc_dfs = [data_dict['acdc_df'] for data_dict in posData.allData_li]
        annotated_frames = lineage_tree_annotated_frames(acdc_dfs)
        if not any(has_lineage_tree_values(acdc_dfs[i]) for i in annotated_frames):
            return -1

        return annotated_frames[-1]

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

        if not force and self.lineage_tree is not None:
            return
        
        mode = str(self.modeComboBox.currentText())
        if mode != 'Normal division: Lineage tree' and not force:
            return

        posData = self.data[self.pos_i]
        last_tracked_i = self.get_last_tracked_i()
        defaultMode = 'Viewer'
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
            msg = widgets_message_box.myMessageBox()
            msg.critical(
                self, 'Tracking was never checked', txt
            )
            self.cancelModeInit(defaultMode)
            return

        proceed = True
        # Last annotated frame index (0 if no frame is annotated), with the
        # same criterion as the lineage tree's `frames_for_dfs`
        last_lin_tree_frame_i = max(
            lineage_tree_annotated_frames(
                data_dict['acdc_df'] for data_dict in posData.allData_li
            ),
            default=0
        )

        if last_lin_tree_frame_i == 0:
            # Remove undoable actions from segmentation mode
            posData.UndoRedoStates[0] = []
            self.undoAction.setEnabled(False)
            self.redoAction.setEnabled(False)

        if posData.frame_i > last_lin_tree_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_lin_tree_frame_i+1}.<br><br>
                Do you want to restart lineage tree analysis from frame 
                {last_lin_tree_frame_i+1}?<br>
            """)
            _, yesButton, stayButton = msg.warning(
                self, 'Go to last annotated frame?', txt, 
                buttonsTexts=(
                    'Cancel', f'Yes, go to frame {last_lin_tree_frame_i+1}', 
                    'No, stay on current frame')
            )
            if yesButton == msg.clickedButton:
                msg = 'Looking good!'
                self.last_lin_tree_frame_i = last_lin_tree_frame_i
                posData.frame_i = last_lin_tree_frame_i
                self.setLookingGoodTitle()
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
                self.cancelModeInit(defaultMode)
                proceed = False
                return
            
        elif posData.frame_i < last_lin_tree_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_lin_tree_frame_i+1}.<br><br>
                Do you want to restart lineage tree analysis from frame
                {last_lin_tree_frame_i+1}?<br>
            """)
            goTo_last_annotated_frame_i = msg.question(
                self, 'Go to last annotated frame?', txt, 
                buttonsTexts=('Yes', 'No', 'Cancel')
            )[0]
            if goTo_last_annotated_frame_i == msg.clickedButton:
                msg = 'Looking good!'
                self.setLookingGoodTitle()
                self.last_lin_tree_frame_i = last_lin_tree_frame_i
                posData.frame_i = last_lin_tree_frame_i
                self.get_data(lin_tree_init=False)
                self.updateAllImages() # i dont think I need to change this
                self.updateScrollbars() # i dont think I need to change this
            elif msg.cancel:
                msg = 'Lineage tree analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.cancelModeInit(defaultMode)
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
            self.lineage_tree = normal_division_lineage_tree(gui=self)

            msg = 'Lineage tree analysis initialized!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)

        return proceed

    @disableWindow
    def propagateLinTreeAction(self, dummy_for_button=None):
        """
        Propagates the lineage tree based on the current frame_i. Used in self.propagateLinTreeButton.

        Nothing to propagate without a lineage tree or when the current
        frame has no lineage tree annotations (e.g. reset after a
        segmentation edit and not annotated again yet, see
        `resetLin_tree_future`).
        """
        posData = self.data[self.pos_i]
        df = posData.allData_li[posData.frame_i]['acdc_df']
        if (
                self.lineage_tree is None
                or not has_lineage_tree_values(df)
                or 'parent_ID_tree' not in df.columns
            ):
            self.logger.info(
                f'Frame n. {posData.frame_i+1} has no lineage tree '
                'annotations: nothing to propagate.'
            )
            return

        self.lineage_tree.propagate(posData.frame_i)
        if posData.frame_i == self.original_df_lin_tree_i:
            self.original_df_lin_tree = posData.allData_li[posData.frame_i]['acdc_df'].copy()

        self.logger.info('Lineage tree propagated.')

    def resetLin_tree_future(self, from_frame_i=None):
        """Remove the lineage tree annotations from frame `from_frame_i` to
        the end, from everywhere they could come back from:

        * Visited frames (`allData_li`): their tables lose the lineage tree
          columns (`lineage_tree_columns`, including the extra
          `sister_ID_tree_<i>`) and, in frames without cell cycle
          annotations (no `cell_cycle_stage` column), the cell cycle
          columns that the lineage tree also writes
          (`LINEAGE_TREE_SHARED_CCA_COLS`).
        * The table loaded from file (`posData.acdc_df`, read by `get_data`
          for frames not visited yet, also after `reInitLastSegmFrame`):
          the same columns are set to NaN in the rows of these frames,
          shared columns only in rows without `cell_cycle_stage`. Rows are
          kept, they hold the other data of the frames.
        * The lineage tree of "Normal division: Lineage tree" mode
          (`self.lineage_tree`): the frames leave `frames_for_dfs` and the
          entries of the per-ID caches `first_frame_i_for_ID` and
          `ID_frame_i_lookup` that point to them are dropped (the rows
          there have no lineage values left to read; the tree finds the
          IDs again by searching the frames). `max_daughters_added` is
          kept, the earlier frames still have those sister columns. The
          reset frames, including the current one, are annotated again
          when the next frame is entered with Next (`autoLinTree_df`
          annotates the missing frames before it first): the current
          frame's masks may still change after the warning. When frame 0
          is reset no annotated frame is left: in lineage tree mode the
          tree is created again from frame 0 (every object a root, as when
          the mode starts), elsewhere it is dropped. In lineage tree mode
          the frames scrollbar maximum is updated.
        * The backup of the lineage tree edits (`original_df_lin_tree`,
          see `repeat_click_and_backup`) if it is of one of these frames.

        Nothing is stored: the tables are edited in place.

        Parameters
        ----------
        from_frame_i : int, optional
            First frame to reset. Default is the current frame.
        """
        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i

        for i in range(from_frame_i, posData.SizeT):
            df = posData.allData_li[i]['acdc_df']
            if df is None:
                continue
            cols = lineage_tree_columns(df)
            if 'cell_cycle_stage' not in df.columns:
                cols += [
                    col for col in LINEAGE_TREE_SHARED_CCA_COLS
                    if col in df.columns
                ]
            if cols:
                posData.allData_li[i]['acdc_df'] = df.drop(columns=cols)

        self._blankLinTreeLoadedAcdcDf(posData, from_frame_i)

        if self.lineage_tree is not None:
            mode = str(self.modeComboBox.currentText())
            isLinTreeMode = mode == 'Normal division: Lineage tree'
            if from_frame_i > 0:
                self._removeLinTreeFrames(from_frame_i)
            elif isLinTreeMode and posData.allData_li[0]['acdc_df'] is not None:
                # The mode's actions (e.g. Propagate) need a tree: start
                # it again from frame 0, as `initLinTree` does
                self.lineage_tree = normal_division_lineage_tree(gui=self)
            else:
                self.lineage_tree = None

            if isLinTreeMode:
                self.setNavigateScrollBarMaximum()

        backup_frame_i = self.original_df_lin_tree_i
        if backup_frame_i is not None and backup_frame_i >= from_frame_i:
            self.clearLineageTreeBackup()

    def _blankLinTreeLoadedAcdcDf(self, posData, from_frame_i):
        # Lineage tree values of frames >= from_frame_i in the table loaded
        # from file to NaN (see `resetLin_tree_future`)
        acdc_df = posData.acdc_df
        if acdc_df is None:
            return

        reset_rows = acdc_df.index.get_level_values(0) >= from_frame_i
        if not reset_rows.any():
            return

        lineage_cols = lineage_tree_columns(acdc_df)
        shared_cols = [
            col for col in LINEAGE_TREE_SHARED_CCA_COLS
            if col in acdc_df.columns
        ]
        if not lineage_cols and not shared_cols:
            return

        if 'cell_cycle_stage' in acdc_df.columns:
            no_cca_rows = acdc_df['cell_cycle_stage'].isna().to_numpy()
            shared_cols_rows = reset_rows & no_cca_rows
        else:
            shared_cols_rows = reset_rows

        # Shallow copy: the replaced columns do not change the old table
        acdc_df = acdc_df.copy(deep=False)
        for col in lineage_cols:
            acdc_df[col] = acdc_df[col].mask(reset_rows)
        for col in shared_cols:
            acdc_df[col] = acdc_df[col].mask(shared_cols_rows)
        posData.acdc_df = acdc_df

    def _removeLinTreeFrames(self, from_frame_i):
        # Frames >= from_frame_i are not annotated anymore in the lineage
        # tree object (see `resetLin_tree_future`)
        lineage_tree = self.lineage_tree
        lineage_tree.frames_for_dfs.difference_update([
            i for i in lineage_tree.frames_for_dfs if i >= from_frame_i
        ])
        for lookup in (
                lineage_tree.first_frame_i_for_ID,
                lineage_tree.ID_frame_i_lookup
            ):
            stale_IDs = [
                ID for ID, frame_i in lookup.items() if frame_i >= from_frame_i
            ]
            for ID in stale_IDs:
                del lookup[ID]

    def setAnnotOptionsLin_treeMode(self):
        self.storeAnnotOptionsBeforeModeSwitch()
        # See comment in `setAnnotOptionsCcaMode`
        self.annotIDsCheckbox(0).setCheckedAlwaysEmit(False)
        self.annotLineageInfoCheckbox(0).setCheckedAlwaysEmit(True)
        self.annotMotherDaughterLineCheckbox(0).setCheckedAlwaysEmit(True)
