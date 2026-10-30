from typing import Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel, Field

_Model = TypeVar('_Model', bound=BaseModel)


def parse_response(model_cls: Type[_Model], data: Any) -> _Model:
    """Validate `data` as `model_cls` with Pydantic 2 or Pydantic 1.

    Pydantic is not pinned, because the main Cell-ACDC environment may share
    it with other packages that still use version 1.
    """
    if hasattr(model_cls, 'model_validate'):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


class HealthResponse(BaseModel):
    status: str = "ok"
    model: Optional[str] = None

class Capabilities(BaseModel):
    segment: bool = True
    segment3DT: bool = False
    second_ch_img_to_stack: bool = True
    predictCcaState: bool = False

class InfoResponse(BaseModel):
    name: str
    version: str = "unknown"
    gpu: bool = False
    capabilities: Capabilities = Field(default_factory=Capabilities)
    init_params: Dict[str, Any] = Field(default_factory=dict)
    segment_params: Dict[str, Any] = Field(default_factory=dict)

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    progress: float = 0.0
    error: Optional[str] = None
