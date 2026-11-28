"""Widget-free per-position and per-frame data of the measurements viewer:
the PositionDataUtils mixin (a base of MeasurementsViewerUtils in
measurements_viewer/viewer_utils.py), the frame masking helpers
(analysis_last_frame, mask_frames), the acdc_output file stamp and the
intensity rescaling texts. Imports no module that imports it (it is a
base-class module of viewer_utils.py). Spec:
measurements_viewer/__init__.py.
"""
import copy
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
from cellacdc.myutils import table_utils
from cellacdc import core, settings_csv_path
from cellacdc.timepoint_predictors import parse_flags
from . import annotation_data as viewer_annotation_data


RESCALE_INTENS_HOW_TEXTS = (
    'Rescale each 2D image',
    'Rescale across z-stack',
    'Rescale across time frames',
    'Do no rescale, display raw image'
)


def analysis_last_frame(posData, mask=True):
    """Last frame index that the calculations of `posData` use, or None
    to use all frames: ``posData.lastAnalyzedFrame`` when `mask` is True
    and the position is not analyzed until its last frame."""
    if not mask:
        return None
    last = getattr(posData, 'lastAnalyzedFrame', None)
    SizeT = getattr(posData, 'SizeT', None)
    if last is None or SizeT is None or last >= SizeT - 1:
        return None
    return int(last)


def mask_frames(df, last_frame):
    """Rows of `df` (``frame_i`` index level or column) up to `last_frame`
    included, or `df` itself (no copy) when `last_frame` is None."""
    if last_frame is None or df is None:
        return df
    if 'frame_i' in df.columns:
        frames = df['frame_i'].to_numpy()
    else:
        frames = df.index.get_level_values('frame_i').to_numpy()
    return df[frames <= last_frame]


def acdc_df_file_stamp(filepath):
    """(modification time, size) of an acdc_output file, None if missing.
    Used to detect files changed on disk (e.g., by the main GUI) since they
    were loaded or saved by the viewer."""
    if not filepath:
        return None
    try:
        stat = os.stat(filepath)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class PositionDataUtils:
    """Widget-free access to the per-position and per-frame data of the
    measurements viewer, for the window and the shared GUI mixins:
    frame_i, posData, get_data, getAcdcDf/getFrameAcdcDf,
    initPosAnnotData, analysis frames, labels and IDs, lineage-tree
    stand-ins, unsaved/changed-on-disk table state and settings. Must
    not call UI methods. Base of MeasurementsViewerUtils
    (measurements_viewer/viewer_utils.py).
    """

    def viewerManagedColumns(self):
        return {
            *(self.timepointColumn(identifier)
              for identifier in self.timepointIdentifiers),
            *self.customAnnotNames(), viewer_annotation_data.VISITED_COLUMN
        }

    def hasUnsavedAcdcDf(self, posData):
        if not hasattr(posData, 'savedAcdcDf'):
            return False
        current = getattr(posData, 'acdc_df', None)
        if current is None or current.empty:
            return False
        saved = posData.savedAcdcDf
        if saved is None:
            return True
        current = self._standardizeAcdcDf(current)
        # Empty viewer columns (no marks yet) are not changes to the file.
        managed = self.viewerManagedColumns()
        def unmarked(df, other):
            return [
                column for column in managed
                if column in df and column not in other
                and df[column].fillna(0).eq(0).all()
            ]
        current, saved = (
            current.drop(columns=unmarked(current, saved)),
            saved.drop(columns=unmarked(saved, current))
        )
        try:
            pd.testing.assert_frame_equal(
                current, saved, check_dtype=False, check_like=True,
                check_index_type=False, check_column_type=False
            )
        except AssertionError:
            return True
        return False

    def unsavedAcdcDfPositions(self):
        return [
            posData for posData in self.data or []
            if self.hasUnsavedAcdcDf(posData)
        ]

    def isAcdcDfChangedOnDisk(self, posData):
        """True if the acdc_output file of `posData` was modified (e.g., saved
        by the main GUI) since the viewer loaded, reloaded or saved it."""
        if not hasattr(posData, 'acdcDfFileStamp'):
            return False
        filepath = getattr(posData, 'acdc_output_csv_path', None)
        return acdc_df_file_stamp(filepath) != posData.acdcDfFileStamp

    @property
    def frame_i(self):
        # The current frame is stored in the current position data as in 
        # guiWin (e.g., `SearchIDGuiElements.findID` sets `posData.frame_i`)
        if not self.data:
            return 0
        return self.data[self.pos_i].frame_i

    @frame_i.setter
    def frame_i(self, value):
        if not self.data:
            return
        self.data[self.pos_i].frame_i = value

    def loadSettings(self):
        self.settings_csv_path = settings_csv_path
        if os.path.exists(settings_csv_path):
            self.df_settings = pd.read_csv(
                settings_csv_path, index_col='setting'
            )
        else:
            self.df_settings = pd.DataFrame(
                {'setting': [], 'value': []}, dtype=object
            ).set_index('setting')
        
        if 'is_bw_inverted' not in self.df_settings.index:
            self.df_settings.at['is_bw_inverted', 'value'] = 'No'
        
        try:
            self.defaultRescaleIntensHow = (
                self.df_settings.at['default_rescale_intens_how', 'value']
            )
        except Exception as err:
            self.defaultRescaleIntensHow = RESCALE_INTENS_HOW_TEXTS[0]

    def hasCcaAnnotations(self):
        return any(
            getattr(posData, 'cca_df', None) is not None 
            or (
                getattr(posData, 'acdc_df', None) is not None
                and 'cell_cycle_stage' in posData.acdc_df.columns
            )
            for posData in self.data
        )

    def setPosDataDefaults(self):
        """Add the metadata attributes that `load.loadData` always sets, but
        that minimal posData objects passed to `setData` might lack"""
        defaults = {
            'PhysicalSizeX': 1.0, 'PhysicalSizeY': 1.0, 'PhysicalSizeZ': 1.0,
            'TimeIncrement': 1.0, 'basename': '', 'loadedFluoChannels': set(),
        }
        for posData in self.data:
            for name, value in defaults.items():
                if not hasattr(posData, name):
                    setattr(posData, name, copy.copy(value))
            for name in ('fluo_data_dict', 'fluo_bkgrData_dict'):
                if not hasattr(posData, name):
                    setattr(posData, name, {})

    def get_labels_array(
            self, from_store=False, frame_i=None, return_existing=False,
            return_copy=True
        ):
        posData = self.posData()
        if frame_i is None:
            frame_i = posData.frame_i
        lab = posData.segm_data[frame_i]
        if return_copy:
            lab = lab.copy()
        if return_existing:
            return lab, False
        return lab

    def posData(self):
        return self.data[self.pos_i]

    def initPosAnnotData(self, posData):
        """Initialize the per-frame stored data (`allData_li`) and the 
        attributes of `posData` required by the annotations as in 
        `guiWin` and return the set of all IDs.

        Uses the regionprops computed in the loading worker thread
        (`precompute_position_regionprops`), if any. The rest runs on the
        GUI thread (e.g. `loadTrackedLostCentroids` may show a dialog)."""
        if getattr(posData, '_isViewerAnnotDataInit', False):
            return posData.allIDs

        precomputed = getattr(posData, '_viewerPrecomputedRps', None)
        if precomputed is not None:
            del posData._viewerPrecomputedRps
        if (isinstance(precomputed, tuple)
                and precomputed[0] is posData.segm_data):
            _, posData.allData_li, allIDs = precomputed
        else:
            posData.allData_li = [None]*posData.SizeT
            allIDs, posData = core.count_objects_and_init_rps(
                posData, self.logger.info
            )
        for i in range(posData.SizeT):
            if posData.allData_li[i] is None:
                posData.allData_li[i] = table_utils.get_empty_stored_data_dict()
        posData.allIDs = set(allIDs)
        
        try:
            posData.loadTrackedLostCentroids()
        except Exception as err:
            pass
        tracked_lost_centroids = getattr(posData, 'tracked_lost_centroids', None)
        if not isinstance(tracked_lost_centroids, dict):
            posData.tracked_lost_centroids = {}
        for i in range(posData.SizeT):
            posData.tracked_lost_centroids.setdefault(i, set())
        
        posData.lost_IDs = []
        posData.new_IDs = []
        posData.old_IDs = []
        posData.trackedLostIDs = set()
        posData.whitelist = getattr(posData, 'whitelist', None)
        posData.cca_df = None
        SizeY, SizeX = posData.segm_data.shape[-2:]
        if not hasattr(posData, 'SizeY'):
            posData.SizeY = SizeY
        if not hasattr(posData, 'SizeX'):
            posData.SizeX = SizeX
        posData._isViewerAnnotDataInit = True
        return posData.allIDs

    def getFrameAcdcDf(self, frame_i):
        """Get the acdc_df of `frame_i` (index is Cell_ID) as stored in 
        `posData.allData_li` by `guiWin`"""
        if self.acdc_df is None:
            return
        
        frames = self.acdc_df.index.get_level_values(0)
        if frame_i not in frames:
            return
        
        df = self.acdc_df.loc[frame_i].copy()
        if 'cell_cycle_stage' in df.columns:
            cca_cols = df.columns.intersection(self.cca_df_colnames)
            cca_df = df[cca_cols].dropna()
            if cca_df.empty:
                df = df.drop(columns=self.cca_df_colnames, errors='ignore')
            else:
                df = df.loc[cca_df.index]
                cols = df.columns.intersection(self.cca_df_int_cols)
                try:
                    df[cols] = df[cols].astype('Int64')
                except Exception as err:
                    pass
        return df

    def getBaseCca_df(self, with_tree_cols=False):
        """Read-only stand-in for `guiWin.getBaseCca_df`: an empty cell
        cycle table (no default annotations are invented for frames that
        were not annotated)."""
        return pd.DataFrame(columns=self.cca_df_colnames).rename_axis(
            'Cell_ID'
        )

    def get_data(self):
        """Load data of the current frame into `posData` as in
        `guiWin.get_data` (read-only)"""
        posData = self.posData()
        frame_i = posData.frame_i
        storedData = posData.allData_li[frame_i]
        posData.lab = self.getLabels(frame_i)
        rp = storedData['regionprops']
        if rp is None:
            rp = self._acdcRegionProps(posData.lab, precache_centroids=False)
            storedData['regionprops'] = rp
        storedData['labels'] = posData.lab
        posData.rp = rp
        posData.IDs = rp.IDs
        
        df = self.getFrameAcdcDf(frame_i)
        storedData['acdc_df'] = df
        cca_df = None
        if df is not None and 'cell_cycle_stage' in df.columns:
            cca_cols = df.columns.intersection(self.cca_df_colnames)
            cca_df = df[cca_cols].copy().dropna()
        posData.cca_df = cca_df

    def getAcdcDf(self, posData=None):
        """Get the acdc_df of the position indexed by (frame_i, Cell_ID)."""
        if posData is None:
            posData = self.posData()
        
        acdc_df = getattr(posData, 'acdc_df', None)
        if acdc_df is None or acdc_df.empty:
            return
        
        acdc_df = self._standardizeAcdcDf(acdc_df).sort_index()
        self.ensureTimepointColumns(acdc_df)
        self.ensureCustomAnnotColumns(acdc_df)
        if viewer_annotation_data.VISITED_COLUMN not in acdc_df:
            acdc_df[viewer_annotation_data.VISITED_COLUMN] = False
        else:
            acdc_df[viewer_annotation_data.VISITED_COLUMN] = parse_flags(acdc_df[viewer_annotation_data.VISITED_COLUMN])
        # The acdc_df of the position is the only store of the time points
        posData.acdc_df = acdc_df
        return acdc_df

    def analysisLastFrame(self, posData=None):
        """Last frame index used by the calculations (predictors, phases,
        measurement processing, combined measurements and conditional
        formatting) of a position, or None when they use all frames
        (`analysis_last_frame` with the "Ignore frames after the last
        analyzed frame" option)."""
        if posData is None:
            posData = self.posData()
        return analysis_last_frame(posData, self.maskUnanalyzedFrames)

    def analysisSizeT(self, posData=None):
        """Number of frames seen by the calculations of a position (the
        ``SizeT`` of equations): the frames up to `analysisLastFrame`, else
        ``posData.SizeT``."""
        if posData is None:
            posData = self.posData()
        last = self.analysisLastFrame(posData)
        if last is not None:
            return last + 1
        return getattr(posData, 'SizeT', None)

    def analysisDf(self, acdc_df, posData=None):
        """Rows of `acdc_df` (a table of `posData`) used by the
        calculations: up to `analysisLastFrame`, or `acdc_df` itself."""
        if posData is None:
            posData = self.posData()
        return mask_frames(acdc_df, self.analysisLastFrame(posData))

    def dialogAnalysisDf(self, acdc_df, posData):
        """`analysisDf` for the editor dialogs (`posData` may be None)."""
        if posData is None or acdc_df is None:
            return acdc_df
        return self.analysisDf(acdc_df, posData)

    def dialogAnalysisSizeT(self, posData):
        """`analysisSizeT` for the editor dialogs (`posData` may be None)."""
        if posData is None:
            return None
        return self.analysisSizeT(posData)

    def allAcdcDfs(self):
        """Unique tables of the loaded positions including the displayed one."""
        dfs = [getattr(posData, 'acdc_df', None) for posData in self.data or []]
        dfs.append(self.acdc_df)
        unique = {id(df): df for df in dfs if df is not None}
        return list(unique.values())

    def cellHasRow(self, ID, frame_i, posData=None):
        acdc_df = self.timepointsDf(posData)
        return acdc_df is not None and (frame_i, ID) in acdc_df.index

    def initLineageTree(self):
        """Stand-in for the `normal_division_lineage_tree` of `guiWin` 
        (only `frames_for_dfs` is required by 
        `AnnotationsGuiElements.drawAllLineageTreeLines`). Lineage tree lines 
        are drawn from the saved `parent_ID_tree` column of the acdc_df."""
        self.lineage_tree = None
        if self.acdc_df is None:
            return
        
        if 'parent_ID_tree' not in self.acdc_df.columns:
            return
        
        frames = self.acdc_df.index.get_level_values(0).unique()
        self.lineage_tree = SimpleNamespace(frames_for_dfs=set(frames))

    def _getLineageTreeFrameDf(self, frame_i, rp, prev_IDs=None):
        df = self.getFrameAcdcDf(frame_i)
        if df is None:
            return
        
        # Parents that cannot be drawn are unknown (-1) as in `guiWin`
        parent_IDs = df['parent_ID_tree'].fillna(-1).astype(int)
        isDrawable = df.index.isin(rp.IDs)
        if prev_IDs is not None:
            isDrawable = isDrawable & parent_IDs.isin(prev_IDs).to_numpy()
        df['parent_ID_tree'] = parent_IDs.where(isDrawable, -1)
        return df

    def initLineageTreeFrameData(self):
        """Store the acdc_df of current and previous frame and the
        regionprops of the previous frame (never visited frames have none)
        required by `AnnotationsGuiElements.drawAllLineageTreeLines` in
        `posData.allData_li`. Returns False if the tables were not stored
        (`drawAllLineageTreeLines` then uses the tables stored by
        `get_data`, if any)."""
        if self.lineage_tree is None:
            return False
        
        posData = self.posData()
        frame_i = posData.frame_i
        if frame_i == 0:
            return True
        
        prevData = posData.allData_li[frame_i-1]
        prev_rp = prevData['regionprops']
        if prev_rp is None:
            prev_lab = self.getLabels(frame_i-1)
            prev_rp = self._acdcRegionProps(prev_lab, precache_centroids=False)
            prevData['regionprops'] = prev_rp
        
        prev_df = self._getLineageTreeFrameDf(frame_i-1, prev_rp)
        df = self._getLineageTreeFrameDf(
            frame_i, posData.rp, prev_IDs=prev_rp.IDs
        )
        if prev_df is None or df is None:
            return False
        
        prevData['acdc_df'] = prev_df
        posData.allData_li[frame_i]['acdc_df'] = df
        return True

    def getAllIDs(self):
        if self.acdc_df is not None:
            IDs = self.acdc_df.index.get_level_values('Cell_ID').unique()
            return sorted([int(ID) for ID in IDs if ID > 0])
        
        posData = self.posData()
        IDs = set()
        for frame_i in range(posData.SizeT):
            IDs.update(np.unique(self.getLabels(frame_i)))
        IDs.discard(0)
        return sorted([int(ID) for ID in IDs])

    def getLabels(self, frame_i=None):
        """Get labels of `frame_i` (default current frame)."""
        posData = self.posData()
        if frame_i is None:
            frame_i = self.frame_i
        
        key = (self.pos_i, frame_i)
        if key == self._labKey:
            return self._lab
        
        lab = posData.segm_data[frame_i]
        
        if frame_i == self.frame_i:
            self._lab, self._labKey = lab, key
        return lab

    def getIDframes(self, ID):
        """Get the sorted frame indices where `ID` is present."""
        key = (self.pos_i, ID)
        cached = self._IDframesCache.get(key)
        if cached is not None:
            return cached
        
        frames = None
        if self.acdc_df is not None:
            try:
                df_ID = self.acdc_df.xs(ID, level='Cell_ID')
                frames = df_ID.index.get_level_values(0).unique().to_numpy()
            except KeyError:
                frames = None
        
        if frames is None or len(frames) == 0:
            segm_data = self.posData().segm_data
            frames = np.array([
                frame_i for frame_i in range(len(segm_data))
                if np.any(segm_data[frame_i] == ID)
            ], dtype=int)
        
        frames = np.sort(frames.astype(int))
        self._IDframesCache[key] = frames
        return frames

    def positionTables(self, posData):
        """The tables of `posData` (its acdc_df and, for the displayed
        position, the working table)."""
        tables = [getattr(posData, 'acdc_df', None)]
        if self.data and posData is self.posData():
            tables.append(self.acdc_df)
        unique = {id(table): table for table in tables if table is not None}
        return list(unique.values())

    def clipFrame(self, frame_i):
        return int(min(max(frame_i, 0), self.posData().SizeT-1))

    def getMeasurementsDf(self):
        """Get the subset of acdc_df for the selected ID (index is frame_i)"""
        if self.acdc_df is None or self.selectedID is None:
            return
        
        try:
            df_ID = self.acdc_df.xs(self.selectedID, level='Cell_ID')
        except KeyError:
            return
        
        return df_ID.sort_index()

    @staticmethod
    def _normalize(values):
        finite = np.isfinite(values)
        normalized = np.full(values.shape, np.nan, dtype=float)
        if not finite.any():
            return normalized
        _min, _max = values[finite].min(), values[finite].max()
        if _max == _min:
            normalized[finite] = 0
        else:
            normalized[finite] = (values[finite] - _min) / (_max - _min)
        return normalized

    def update_rp(self, *args, **kwargs):
        self.get_data()
