from dataclasses import dataclass
from typing import Any


@dataclass
class BaseQuery:
    def to_dict(self):
        raise NotImplementedError("Subclasses should implement this!")


@dataclass
class Match(BaseQuery):
    field: str
    value: Any

    def to_dict(self):
        return {"match": {self.field: self.value}}


@dataclass
class MatchPhrase(Match):
    def to_dict(self):
        return {"match_phrase": {self.field: self.value}}


@dataclass
class MatchAll(BaseQuery):
    def to_dict(self):
        return {"match_all": {}}


@dataclass
class Bool(BaseQuery):
    must: list[BaseQuery]

    def to_dict(self):
        return {"bool": {"must": [q.to_dict() for q in self.must]}}
