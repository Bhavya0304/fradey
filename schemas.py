from pydantic import BaseModel
from typing import Optional, Dict

class HandshakeResponse(BaseModel):
    session_id: str
    crypto: Dict[str, str]

class ControlEvent(BaseModel):
    session_id: str
    type: str  # "DTMF", "END", "MENU", "BARge_IN", etc.
    data: Optional[Dict] = None
