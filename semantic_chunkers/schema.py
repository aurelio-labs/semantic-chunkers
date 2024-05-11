from typing import List, Optional

from pydantic.v1 import BaseModel


class ChunkSet(BaseModel):
    docs: List[str]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None

    @property
    def content(self) -> str:
        return " ".join(self.docs)
