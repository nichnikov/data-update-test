from pydantic import BaseModel


class SearchRequest(BaseModel):
    pubid: int
    chat_id: int | None = None
    userid: int = 0
    text: str


class SearchResponse(BaseModel):
    templateId: int = 0
    templateText: str = ""
    etalon_text: str = ""
    algorithm: str = ""
    score: float = 0.0
