from abc import ABC, abstractmethod

from pymystem3 import Mystem

from core.elastic.client import ElasticClient
from core.schemas import SearchResponse
from core.text_preprocessing.lemmatizer import TextLemmatizer


class Classifier(ABC):
    def __init__(self, es_client: ElasticClient, mystem: Mystem, params):
        self.es_client = es_client
        self.params = params

        self.lemmatizer = TextLemmatizer(mystem=mystem)
        self.lemmatizer.add_stopwords(stopwords=self.params.stopwords)

    @abstractmethod
    async def classify(self, text: str, pub_id: int) -> SearchResponse:
        pass


class ModelMixin(ABC):
    models_names: list[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models: dict = {}

    @abstractmethod
    def load_models(self):
        pass


class ClassifierWithModel(ModelMixin, Classifier, ABC):
    pass
