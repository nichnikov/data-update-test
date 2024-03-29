import contextlib
import logging
import re

from core.classifiers.base import Classifier
from core.elastic.queries import Bool, Match, MatchPhrase
from core.exceptions import AnswerNotFound, ESResponseEmpty
from core.schemas import SearchResponse

logger = logging.getLogger(__name__)


class KosguClassifier(Classifier):
    @staticmethod
    def score(lem_input_text: str, etalon: str) -> bool:
        """Поиск по особым правилам, (для косгу)
        выбирается самый длинный эталон, входящий в исходный запрос
        special_patterns must be lematized"""
        if re.findall(etalon, lem_input_text):
            logger.info("KosguClassifier found %s in %s", etalon, lem_input_text)
            return True
        else:
            return False

    async def classify(self, text: str, pub_id: int) -> SearchResponse:

        special_patterns = "косг|квр"

        tokens_str = " ".join(self.lemmatizer.tokenization([text])[0])
        serching_text = re.sub(special_patterns, "", tokens_str)
        etalons_search_result = await self.es_client.q_search(
            index=self.params.es_clusters_index,
            query=Bool([MatchPhrase("Topic", "КОСГУ робот"), Match("LemCluster", serching_text)]),
        )

        # удаление special_patterns из найденных эталонов и сортировка по длине:
        etalons_search_result_sorted = sorted(
            [
                (
                    d["ID"],
                    re.sub(special_patterns, "", d["LemCluster"]),
                    len(re.sub(special_patterns, "", d["LemCluster"]).split()),
                    d["ParentPubList"],
                )
                for d in etalons_search_result
            ],
            key=lambda x: x[2],
            reverse=True,
        )

        for id, et, ln, pbs in etalons_search_result_sorted:
            if pub_id not in pbs:
                continue

            score = self.score(re.sub(special_patterns, "", tokens_str), et)
            if not score:
                continue

            with contextlib.suppress(ESResponseEmpty):
                answers_search_result = await self.es_client.q_search(
                    index=self.params.es_answers_index,
                    query=Bool([MatchPhrase("templateId", id), MatchPhrase("pubId", pub_id)]),
                )
            if not answers_search_result:
                continue

            return SearchResponse(
                templateId=answers_search_result[0]["templateId"],
                templateText=answers_search_result[0]["templateText"],
                etalon_text=et,
                algorithm="Kosgu",
                score=score,
            )

        raise AnswerNotFound(f"didn't find anything for text '''{tokens_str}'''")
