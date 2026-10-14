from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    model: Optional[str] = None

class Capabilities(BaseModel):
    segment: bool = True
    segment3DT: bool = False
    second_ch_img_to_stack: bool = True

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
