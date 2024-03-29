import contextlib
import logging

from core.classifiers.base import Classifier
from core.elastic.queries import Bool, Match, MatchPhrase
from core.exceptions import AnswerNotFound, ESResponseEmpty
from core.schemas import SearchResponse

logger = logging.getLogger(__name__)


class JaccardClassifier(Classifier):
    @staticmethod
    def score(text1: str, text2: str) -> float:
        """Jaccard similarity score"""
        intersection = set(text1.split()) & set(text2.split())
        union = set(text1.split()).union(set(text2.split()))
        if len(union) != 0:
            return float(len(intersection) / len(union))
        return 0.0

    async def classify(self, text: str, pub_id: int) -> SearchResponse:
        tokens_str = " ".join(self.lemmatizer.tokenization([text])[0])
        etalons_search_result = await self.es_client.q_search(
            index=self.params.es_clusters_index,
            query=Bool([MatchPhrase("ParentPubList", pub_id), Match("LemCluster", tokens_str)]),
        )

        for result in etalons_search_result:
            if pub_id not in result["ParentPubList"]:
                continue
            if (score := self.score(tokens_str, result["LemCluster"])) < self.params.score_threshold:
                continue

            with contextlib.suppress(ESResponseEmpty):
                answers_search_result = await self.es_client.q_search(
                    index=self.params.es_answers_index,
                    query=Bool([Match("templateId", result["ID"]), Match("pubId", pub_id)]),
                )
            if not answers_search_result:
                continue

            return SearchResponse(
                templateId=answers_search_result[0]["templateId"],
                templateText=answers_search_result[0]["templateText"],
                etalon_text=result["Cluster"],
                algorithm="Jaccard",
                score=score,
            )

        raise AnswerNotFound(f"didn't find anything for text '''{tokens_str}'''")
