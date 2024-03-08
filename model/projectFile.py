from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

class ProjectFile(BaseModel):
    filename : str
    file_type : str
    asset_type : str
    file_size : Optional[datetime] = None
    file_path : str
    created_timestamp: Optional[datetime] = None
    project_id : str