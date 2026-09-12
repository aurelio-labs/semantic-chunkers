from typing import List

from pydantic import BaseModel, ConfigDict


class BaseSplitter(BaseModel):
    model_config = ConfigDict(extra="allow")

    def __call__(self, doc: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")
