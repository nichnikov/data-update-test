import logging
import os
import re

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.classifiers.base import ClassifierWithModel
from core.elastic.queries import Bool, Match
from core.exceptions import ScoreTooLow
from core.schemas import SearchResponse
from core.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class SBERTT5Classifier(ClassifierWithModel):
    """Модуль с классификатором, состоящим из Сберта с валидацией Т5"""

    models_names = ["all_sys_paraphrase.transformers", "models_bss", "ruT5-large"]

    def load_models(self):
        self.models["sbert-model"] = SentenceTransformer(
            str(os.path.join(MODELS_DIR, "all_sys_paraphrase.transformers")),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.models["t5-tokenizer"] = T5Tokenizer.from_pretrained(str(os.path.join(MODELS_DIR, "ruT5-large")))
        self.models["t5-model"] = T5ForConditionalGeneration.from_pretrained(
            str(os.path.join(MODELS_DIR, "models_bss"))
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def sbert_ranging(self, lem_query: str, score: float, candidates: list):
        text_emb = self.models["sbert-model"].encode(lem_query)
        ids, ets, lm_ets, answs = zip(*candidates)
        candidate_embs = self.models["sbert-model"].encode(lm_ets)
        scores = cos_sim(text_emb, candidate_embs)
        scores_list = [score.item() for score in scores[0]]
        the_best_result = sorted(list(zip(ids, ets, lm_ets, answs, scores_list)), key=lambda x: x[4], reverse=True)[0]
        logger.info("sbert_ranging the_best_result score = %s", the_best_result[4])

        if the_best_result[4] < score:
            raise ScoreTooLow(
                f"sbert_ranging the_best_result score = {the_best_result[4]} is too low for text {lem_query}"
            )

        return the_best_result

    def t5_validate(self, query: str, answer: str, score: float):
        text = query + " Document: " + answer + " Relevant: "
        input_ids = (
            self.models["t5-tokenizer"]
            .encode(text, return_tensors="pt")
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        outputs = (
            self.models["t5-model"]
            .generate(
                input_ids,
                eos_token_id=self.models["t5-tokenizer"].eos_token_id,
                max_length=64,
                early_stopping=True,
            )
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        outputs_decode = self.models["t5-tokenizer"].decode(outputs[0][1:])
        outputs_logits = self.models["t5-model"].generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=self.models["t5-tokenizer"].eos_token_id,
            max_length=64,
            early_stopping=True,
        )
        sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
        t5_score = sigmoid_0[2].item()
        val_str = re.sub("</s>", "", outputs_decode)
        logger.info("t5_validate answer is %s with score = %s", val_str, t5_score)
        return val_str == "Правда" and t5_score >= score

    async def classify(self, text: str, pub_id: int) -> SearchResponse:
        tokens_str = " ".join(self.lemmatizer.tokenization([text])[0])

        etalons_search_result = await self.es_client.q_search(
            index=self.params.es_clusters_index,
            query=Bool([Match("LemCluster", tokens_str), Match("ParentPubList", pub_id)]),
        )

        results_tuples = [
            (d["ID"], d["Cluster"], d["LemCluster"], d["ShortAnswerText"])
            for d in etalons_search_result[: self.params.num_candidates]
        ]

        sbert_the_best_result = self.sbert_ranging(tokens_str, self.params.model_extra["sbert_score"], results_tuples)

        if not self.t5_validate(tokens_str, sbert_the_best_result[3], self.params.model_extra["t5_score"]):
            raise ScoreTooLow(f"mouse didn't validate answer for input text {text}")

        found_answers = await self.es_client.q_search(
            index=self.params.es_answers_index,
            query=Bool([Match("templateId", sbert_the_best_result[0]), Match("pubId", pub_id)]),
        )

        return SearchResponse(
            templateId=sbert_the_best_result[0],
            templateText=str(found_answers[0]["templateText"]),
            etalon_text=sbert_the_best_result[1],
            algorithm="SbertT5",
            score=sbert_the_best_result[4],
        )
