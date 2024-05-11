from typing import Any, List, Optional

from pydantic.v1 import BaseModel


class ChunkSet(BaseModel):
    docs: List[Any]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None

    @property
    def content(self) -> str:
        return " ".join(self.docs)
