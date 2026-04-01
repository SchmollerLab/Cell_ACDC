from dataclasses import dataclass

@dataclass(frozen=True)
class WfImageDC:
    SizeZ: int = None
    SizeT: int = None
    color: str = 'blue'

    def __str__(self):
        return f'img'
    def info(self):
        return f'img, SizeZ={self.SizeZ}, SizeT={self.SizeT}'


@dataclass(frozen=True)
class WfSegmDC:
    SizeZ: int = None
    SizeT: int = None
    color: str = 'red'

    def __str__(self):
        return f'segm'
    
    def info(self):
        return f'segm, SizeZ={self.SizeZ}, SizeT={self.SizeT}'


@dataclass(frozen=True)
class WfMetricsDC:
    setMetrics: list = None
    color: str = '#2e8b57'

    def __str__(self):
        return f'metrics'
    
    def info(self):
        return f'metrics, set={self.setMetrics}'


def workflow_type_name(type_value):
    """Return canonical type name for workflow data-class instances."""
    if isinstance(type_value, WfImageDC):
        return 'img'
    if isinstance(type_value, WfSegmDC):
        return 'segm'
    if isinstance(type_value, WfMetricsDC):
        return 'metrics'
    return None


def make_workflow_data_class(type_name, SizeZ=None, SizeT=None, setMetrics=None):
    """Create a workflow data class instance from canonical type name."""
    if is_workflow_data_class(type_name):
        type_name = workflow_type_name(type_name)
    if type_name == 'img':
        return WfImageDC(SizeZ=SizeZ, SizeT=SizeT)
    if type_name == 'segm':
        return WfSegmDC(SizeZ=SizeZ, SizeT=SizeT)
    if type_name == 'metrics':
        return WfMetricsDC(setMetrics=setMetrics)
    if type_name == 'any':
        return None
    if type_name is None:
        return None
    
    printl(f"Warning: Unrecognized workflow data type '{type_name}'. Returning None.")
    return None


def is_workflow_data_class(type_value):
    """Return True when value is a supported workflow data-class instance."""
    return isinstance(
        type_value,
        (WfImageDC, WfSegmDC, WfMetricsDC),
    )
