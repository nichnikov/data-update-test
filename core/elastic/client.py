import logging

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.elastic.queries import BaseQuery
from core.exceptions import ESResponseEmpty

logger = logging.getLogger(__name__)


class ElasticSettings(BaseSettings):
    """Elasticsearch settings."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", env_prefix="es_", extra="ignore")
    # model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    hosts: str
    user: str | None = "elastic"
    password: str | None = "changeme"

    max_hits: int = 300
    chunk_size: int = 500

    chat_history_index: str = "chat_history"
    results_index: str = "results"

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Returns basic auth tuple if user and password are specified."""
        if self.user and self.password:
            return self.user, self.password
        return None


class ElasticClient(AsyncElasticsearch):
    """Elasticsearch client."""

    def __init__(self, *args, **kwargs):
        self.conf = ElasticSettings()
        super().__init__(
            hosts=self.conf.hosts,
            basic_auth=self.conf.basic_auth,
            request_timeout=100,
            max_retries=10,
            retry_on_timeout=True,
            *args,
            **kwargs,
        )

    async def create_index(self, index: str) -> None:
        """Creates the index if one does not exist."""
        if not await self.indices.exists(index=index):
            await self.indices.create(index=index)

    async def delete_index(self, index: str) -> None:
        """Deletes the index if one exists."""
        if await self.indices.exists(index=index):
            await self.indices.delete(index=index)

    async def add_docs(self, index_name: str, docs: list[dict]):
        """Adds documents to the index."""
        _gen = ({"_index": index_name, "_source": doc} for doc in docs)
        await async_bulk(self, _gen, chunk_size=self.conf.chunk_size, stats_only=True)
        logger.info("added %i documents to index %s", len(docs), index_name)

    async def q_search(self, index: str, query: BaseQuery, size: int = None) -> list:
        """Searches for query in the index and returns a search result."""

        response = await self.search(
            index=index,
            query=query.to_dict(),
            size=size or self.conf.max_hits,
        )

        if not (hits := response["hits"]["hits"]):
            raise ESResponseEmpty(f"ES didn't find anything for query {query.to_dict()} in {index} index")

        return [
            {
                **d["_source"],
                **{"id": d["_id"]},
                **{"score": d["_score"]},
            }
            for d in hits
        ]

    async def q_delete(self, index: str, query: BaseQuery) -> None:
        """Delete by query."""
        await self.delete_by_query(index=index, query=query.to_dict())
