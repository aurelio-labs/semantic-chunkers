from typing import List

from pydantic.v1 import BaseModel, Extra


class BaseSplitter(BaseModel):
    class Config:
        extra = Extra.allow

    def __call__(self, doc: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")
