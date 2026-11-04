"""Predictors of custom annotations used by the measurements viewer.

A custom annotation is a boolean (0/1) column of the acdc_output table
constant across all rows of a Cell_ID. A predictor evaluates an equation (or
a ``predictor(df)`` function from a Python file) on the rows of each cell and
annotates the entire cell history after consolidating its boolean results.

Unlike the time point predictors (see ``timepoint_predictors``), there is no
time-point selection rule (e.g. "first true"): the result must be
strictly boolean, i.e. one bool per row of the cell or a single bool applying
to all its rows. Numeric results (including 0/1) are rejected. Main-cell
filters, other-cell measurements and per-cell functions work exactly as in
the time point predictors.
"""
from typing import Any, Iterable

import pandas as pd

from .timepoint_predictors import (
    CellFilter, CellTable, OtherCellMeasurements, TimepointPredictor,
    PREDICTOR_NONE, PREDICTOR_EQUATION, PREDICTOR_FILE, PREDICTOR_TYPES,
    SOURCE_CELLS, STRICT_BOOL
)

# Continuation of the time point shortcuts (Q W E R ...) on the next keyboard
# row. H is skipped because it is the zoom out shortcut of the viewer.
DEFAULT_SHORTCUTS = tuple('ASDFGJKL')
CONSOLIDATION_MODES = ('all', 'none', 'most')

PREDICTOR_FILE_TEMPLATE = '''def predictor(df, entire_df=None):
    """df: acdc_df of a single cell (one row per frame, columns include 
    frame_i and Cell_ID).
    entire_df: acdc_df of all cells (e.g., to use the measurements of other 
    cells). Remove the argument if not needed.
    Return one boolean per row of df, consolidated using All/None/Most,
    or a single boolean. The annotation is constant across the cell history."""
    return df['cell_area_pxl'] > 500
'''


class CustomAnnotationPredictor:
    """Validate and execute equation-based or file-based predictors of a 
    custom annotation.

    predict() is the main entry point. It returns 
    ``{Cell_ID: [annotated frame_i, ...]}`` omitting cells without annotated 
    frames. Results must be strictly boolean (see 
    ``CellFilter.strict_boolean_values``).
    """

    @staticmethod
    def validate(predictor: dict[str, Any]) -> None:
        """Validate predictor type, equation and optional filters."""
        if not isinstance(predictor, dict):
            raise ValueError('Invalid custom annotation predictor.')
        kind = predictor.get('type')
        if predictor.get('consolidation', 'all') not in CONSOLIDATION_MODES:
            raise ValueError('Invalid custom annotation consolidation mode.')
        if kind not in PREDICTOR_TYPES:
            raise ValueError(
                f'Invalid custom annotation predictor type: {kind}.'
            )
        if kind == PREDICTOR_EQUATION:
            equation = predictor.get('equation')
            if not isinstance(equation, str) or not equation.strip():
                raise ValueError('The predictor equation cannot be empty.')
            if predictor.get('other_cells') is not None:
                CellFilter.validate_other(predictor['other_cells'])
            if predictor.get('main_cell_filter') is not None:
                CellFilter.validate_main(predictor['main_cell_filter'])
        elif kind == PREDICTOR_FILE:
            filepath = predictor.get('filepath')
            if not isinstance(filepath, str) or not filepath:
                raise ValueError('The predictor file path cannot be empty.')

    @staticmethod
    def describe(predictor: dict[str, Any] | None) -> str:
        """Return the user-facing summary of a predictor configuration."""
        if not predictor:
            predictor = {'type': PREDICTOR_NONE}
        kind = predictor.get('type', PREDICTOR_NONE)
        if kind == PREDICTOR_EQUATION:
            mode = predictor.get('consolidation', 'all').capitalize()
            description = f'{mode}: {predictor["equation"]} (entire cell history)'
            otherCells = predictor.get('other_cells')
            if otherCells is not None:
                settings = CellFilter.validate_other(otherCells)
                description += (
                    f' (uses {SOURCE_CELLS[settings["source"]]}, '
                    f'frames {settings["range_start"]} to '
                    f'{settings["range_end"]}'
                )
                if settings['condition']:
                    description += f', filter: {settings["condition"]}'
                description += ')'
            mainFilter = predictor.get('main_cell_filter')
            if mainFilter is not None:
                settings = CellFilter.validate_main(mainFilter)
                if settings['condition'].strip():
                    description += (
                        f' [cells: {settings["condition_mode"]} of '
                        f'{settings["condition"]}]'
                    )
            return description
        if kind == PREDICTOR_FILE:
            mode = predictor.get('consolidation', 'all').capitalize()
            return f'{mode}: predictor(df) from "{predictor["filepath"]}"'
        return 'None (manual annotation)'

    @staticmethod
    def load_function(filepath: str):
        """Import ``predictor`` from the Python file at ``filepath``."""
        return TimepointPredictor.load_function(filepath)

    @staticmethod
    def consolidate(values, length: int, mode: str = 'all') -> bool:
        """Strict boolean consolidation; Most requires more than half."""
        if mode not in CONSOLIDATION_MODES:
            raise ValueError('Invalid custom annotation consolidation mode.')
        mask = CellFilter.strict_boolean_values(values, length)
        if not length:
            return False
        if mode == 'all':
            return bool(mask.all())
        if mode == 'none':
            return not bool(mask.any())
        return int(mask.sum()) > length / 2

    @staticmethod
    def _annotated_frames(df_ID: pd.DataFrame, values, consolidation) -> list[int]:
        if not CustomAnnotationPredictor.consolidate(
            values, len(df_ID), consolidation
        ):
            return []
        return sorted({int(frame_i) for frame_i in df_ID['frame_i']})

    @staticmethod
    def from_equation(
            acdc_df: pd.DataFrame, equation: str,
            SizeT: int | None = None, IDs: Iterable[int] | None = None,
            other_cells: dict[str, str] | None = None,
            main_cell_filter: dict[str, str] | None = None,
            consolidation: str = 'all'
        ) -> dict[int, list[int]]:
        """Evaluate and consolidate ``equation`` on each cell's rows.

        Steps: (1) optionally reduce IDs to the cells passing the main-cell
        filter; (2) resolve other-cell inputs for the remaining targets;
        (3) evaluate the equation per cell (hence per-cell functions such as
        ``MAX`` can be used). The result must be strictly boolean. With 
        multiple other-cell alternatives, each row's result is True if any
        alternative is True, before cell-wide consolidation.
        Unlike predict(), empty lists are retained
        and frames are not clipped to SizeT.
        """
        IDs = TimepointPredictor.main_cell_filter_IDs(
            acdc_df, main_cell_filter, other_cells=other_cells, SizeT=SizeT,
            IDs=IDs
        )
        acdc_df, equation, alternatives = OtherCellMeasurements.prepare_inputs(
            acdc_df, equation, other_cells, SizeT, IDs=IDs,
            return_alternatives=True
        )
        variables = {'SizeT': SizeT} if SizeT is not None else None
        annotated = {}
        for ID, df_ID in CellTable.iter_cells(acdc_df):
            try:
                values = OtherCellMeasurements.evaluate_with_alternatives(
                    df_ID, equation, alternatives, variables=variables,
                    combine='any', enforce_dtype_output=STRICT_BOOL
                )
                frames = CustomAnnotationPredictor._annotated_frames(
                    df_ID, values, consolidation
                )
            except Exception as err:
                raise ValueError(
                    f'Evaluating the equation failed for Cell_ID {ID}: {err}'
                ) from err
            annotated[ID] = frames
        return annotated

    @staticmethod
    def from_function(
            acdc_df: pd.DataFrame, func, IDs: Iterable[int] | None = None,
            consolidation: str = 'all'
        ) -> dict[int, list[int]]:
        """Call ``func(df)`` on each cell's rows (with ``entire_df`` holding
        all cells, if ``func`` accepts it) and consolidate its strictly
        boolean results into an annotation for the entire cell history."""
        kwargs = {}
        if TimepointPredictor.accepts_entire_df(func):
            kwargs['entire_df'] = (
                CellTable.flatten(acdc_df).sort_values(['Cell_ID', 'frame_i'])
                .reset_index(drop=True)
            )
        annotated = {}
        for ID, df_ID in CellTable.iter_cells(CellTable.subset(acdc_df, IDs)):
            try:
                values = func(df_ID.copy(), **kwargs)
                frames = CustomAnnotationPredictor._annotated_frames(
                    df_ID, values, consolidation
                )
            except Exception as err:
                raise ValueError(
                    f'predictor(df) failed for Cell_ID {ID}: {err}'
                ) from err
            annotated[ID] = frames
        return annotated

    @staticmethod
    def predict(
            acdc_df: pd.DataFrame | None, predictor: dict[str, Any],
            SizeT: int | None = None, IDs: Iterable[int] | None = None
        ) -> dict[int, list[int]]:
        """Predict the annotated frames of each cell.

        If ``IDs`` is not None, only these cells are predicted, while all 
        cells of ``acdc_df`` are available as other cells.

        Returns
        -------
        dict
            ``{Cell_ID: sorted list of annotated frame_i}`` (cells without 
            annotated frames are omitted). Frames outside ``[0, SizeT)`` are 
            discarded.

        Raises
        ------
        ValueError
            Invalid settings, non-boolean results or per-cell 
            equation/callback execution failure.
        """
        CustomAnnotationPredictor.validate(predictor)
        kind = predictor['type']
        if kind == PREDICTOR_NONE or acdc_df is None or acdc_df.empty:
            return {}
        if kind == PREDICTOR_EQUATION:
            annotated = CustomAnnotationPredictor.from_equation(
                acdc_df, predictor['equation'], SizeT=SizeT, IDs=IDs,
                other_cells=predictor.get('other_cells'),
                main_cell_filter=predictor.get('main_cell_filter'),
                consolidation=predictor.get('consolidation', 'all')
            )
        else:
            func = CustomAnnotationPredictor.load_function(
                predictor['filepath']
            )
            annotated = CustomAnnotationPredictor.from_function(
                acdc_df, func, IDs=IDs,
                consolidation=predictor.get('consolidation', 'all')
            )

        predicted = {}
        for ID, frames in annotated.items():
            if SizeT is not None:
                frames = [frame for frame in frames if 0 <= frame < SizeT]
            if frames:
                predicted[ID] = frames
        return predicted
