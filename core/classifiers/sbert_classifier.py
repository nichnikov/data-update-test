import logging
import os

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from core.classifiers.base import ClassifierWithModel
from core.elastic.queries import Bool, Match
from core.exceptions import ScoreTooLow
from core.schemas import SearchResponse
from core.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class SBERTClassifier(ClassifierWithModel):
    models_names = ["all_sys_paraphrase.transformers"]

    def load_models(self):
        self.models["all_sys_paraphrase.transformers"] = SentenceTransformer(
            str(os.path.join(MODELS_DIR, "all_sys_paraphrase.transformers")),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    async def classify(self, text: str, pub_id: int) -> SearchResponse:
        tokens_str = " ".join(self.lemmatizer.tokenization([text])[0])

        etalons_search_result = await self.es_client.q_search(
            index=self.params.es_clusters_index,
            query=Bool([Match("LemCluster", tokens_str), Match("ParentPubList", pub_id)]),
        )

        results_tuples = [
            (d["ID"], d["Cluster"], d["LemCluster"]) for d in etalons_search_result[: self.params.num_candidates]
        ]

        text_emb = self.models["all_sys_paraphrase.transformers"].encode(
            tokens_str, batch_size=64, show_progress_bar=False
        )

        ids, ets, lm_ets = zip(*results_tuples)
        candidate_embs = self.models["all_sys_paraphrase.transformers"].encode(
            lm_ets, batch_size=64, show_progress_bar=False
        )

        scores = cos_sim(text_emb, candidate_embs)
        scores_list = [score.item() for score in scores[0]]

        the_best_result = sorted(list(zip(ids, ets, lm_ets, scores_list)), key=lambda x: x[3], reverse=True)[0]
        logger.info("Best result from BERT: %s", the_best_result)

        if the_best_result[3] < self.params.score_threshold:
            raise ScoreTooLow(f"score {the_best_result[3]} is too low for text '''{tokens_str}'''")

        found_answers = await self.es_client.q_search(
            index=self.params.es_answers_index,
            query=Bool([Match("templateId", the_best_result[0]), Match("pubId", pub_id)]),
        )

        return SearchResponse(
            templateId=the_best_result[0],
            templateText=str(found_answers[0]["templateText"]),
            etalon_text=the_best_result[2],
            algorithm="Sbert",
            score=the_best_result[3],
        )
