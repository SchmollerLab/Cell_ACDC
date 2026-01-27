
import os

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

from ... import _types, myutils, core

from . import get_pretrained_model_names

class AvailableModels:
    values = get_pretrained_model_names()

class AvailableLinkingModes:
    values = ['greedy', 'greedy_nodiv', 'ilp']

class AvailableCellDivisionModes:
    values = ['Normal', 'Asymmetric']

class tracker:
    def __init__(
            self, 
            pretrained_model_name: AvailableModels='general_2d', 
            model_folder_path: _types.FolderPath='', 
            gpu=False
        ) -> None:
        """Initialize tracker

        Parameters
        ----------
        pretrained_model_name : AvailableModels, optional
            Pre-trained model name. Default is 'general_2d'
        model_folder_path : os.PathLike, optional
            Path to the folder containing `config.yaml` file from 
            custom training. Default is ''
        gpu : bool, optional
            If `True`, attempts to try to use the GPU for inference. 
            Default is False
        """        
        device = myutils.get_torch_device()
        if model_folder_path:
            self.model = Trackastra.from_folder(
                model_folder_path, device=str(device)
            )
        else:
            self.model = Trackastra.from_pretrained(
                pretrained_model_name, device=str(device)
            )
    
    def track(
            self, segm_video, video_grayscale, 
            linking_mode: AvailableLinkingModes='greedy', 
            prevent_deleting_objects: bool=True,
            cell_division_mode: AvailableCellDivisionModes='Normal', 
            record_lineage=True
        ):
        """Track the objects in `segm_video`

        Parameters
        ----------
        segm_video : (T, Y, X) np.ndarray of ints
            Input segmentation labels array over time.
        video_grayscale : (T, Y, X) np.ndarray
            Input intensity images over time.
        linking_mode : {'greedy', 'greedy_nodiv', 'ilp'}, optional
            Strategy used to link the predicted associations. Note that 
            'ilp' requires the package `motile`. Default is 'greedy'
        prevent_deleting_objects : bool, optional
            If `True`, prevent Trackastra from removing untracked objects or 
            merging them with other objects. Note that these added objects 
            will not be tracked. Default is `True`.
        cell_division_mode : {'Normal', 'Asymmetric'}, optional
            Type of cell division. `Normal` is the standard cell division, 
            where the mother cell divides into two daughter cells. For the 
            tracking, that means the two daughter cells get a new, unique ID 
            each. Note that division is not detected if 
            `linking_mode == greedy_nodiv`.
            
            `Asymmetric` means that the mother cell grows one daughter 
            cell that eventually divides from the mother (e.g., budding yeast). 
            For the tracking, this means that the mother cell ID keeps 
            existing after division and the daughter cell gets a new, unique ID. 
        record_lineage : bool, optional
            If `True`, store a list of cell lineage annotaions (Cell-ACDC format) 
            in the `self.cca_dfs` list (one DataFrame with index `Cell_ID` per 
            frame). When used through Cell-ACDC, this list will be saved 
            to the acdc_output CSV file. 
        """                
        out = self.model.track(
            video_grayscale, segm_video, mode=linking_mode
        )
        
        try:
            df_ctc, tracked_video = graph_to_ctc(out, segm_video)
        except Exception as e:
            try:
                graph = out[0]
                df_ctc, tracked_video = graph_to_ctc(graph, segm_video)
            except Exception as e2:
                graph = out[1]
                df_ctc, tracked_video = graph_to_ctc(graph, segm_video)
            

        if prevent_deleting_objects:
            tracked_video = core.insert_missing_objects(
                tracked_video, segm_video
            )
        
        if linking_mode == 'greedy_nodiv':
            return tracked_video
        
        acdc_df, cca_dfs, asym_segm_tracked = myutils.df_ctc_to_acdc_df(
            df_ctc, tracked_video, cell_division_mode=cell_division_mode, 
            return_list=True, progressbar=True
        )
        
        if cell_division_mode == 'Asymmetric':
            return asym_segm_tracked
        
        if record_lineage:
            self.cca_dfs = cca_dfs
        
        return tracked_video

    def validate_input(self, segm_video, progress=True):
        import skimage.measure
        warning_text = None
        if progress:
            from tqdm import tqdm
            pbar = tqdm(
                total=len(segm_video), desc='Validating input', unit='frame',
                ncols=100
            )
        
        empty_frames = []
        for frame_i, lab in enumerate(segm_video):
            rp = skimage.measure.regionprops(lab)
            if len(rp) == 0:
                empty_frames.append(frame_i+1)
            
            if progress:
                pbar.update(1)
        
        if empty_frames:
            warning_text = (
                'Trackastra requires that each frame has at least one object.\n\n'
                f'The following frame numbers have no objects:\n\n{empty_frames}'
            )
        
        if progress:
            pbar.close()
        
        return warning_text

def url_help():
    return 'https://github.com/weigertlab/trackastra'