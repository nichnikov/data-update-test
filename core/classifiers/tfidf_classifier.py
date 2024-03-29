import itertools
import logging
import operator
import os
from itertools import chain

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

from core.classifiers.base import ClassifierWithModel
from core.exceptions import ScoreTooLow
from core.schemas import SearchResponse
from core.settings import DATA_DIR

logger = logging.getLogger(__name__)


def group_by_lbs(_l):
    _it = itertools.groupby(_l, operator.itemgetter(0))
    for key, subiter in _it:
        yield key, list(item[1] for item in subiter)


class TFIDFClassifier(ClassifierWithModel):
    index: MatrixSimilarity
    dictionary: Dictionary
    answers: dict

    def load_models(self):
        etalons_df = pd.read_csv(os.path.join(DATA_DIR, "etalons.csv"), sep="\t")
        groups_texts = list(zip(etalons_df["label"], etalons_df["query"]))
        texts_by_groups = sorted(list(group_by_lbs(sorted(groups_texts, key=lambda x: x[0]))), key=lambda x: x[0])

        texts = list(etalons_df["query"])

        tokens = self.lemmatizer.lemmatize_texts(texts)

        dct = Dictionary(tokens)
        texts_by_groups_tokenized = [
            [x for x in chain(*self.lemmatizer.lemmatize_texts(txs))] for grp, txs in texts_by_groups
        ]
        corpus = [dct.doc2bow(item) for item in texts_by_groups_tokenized]

        self.dictionary = dct
        self.models["tfidf"] = TfidfModel(corpus)
        self.index = MatrixSimilarity(self.models["tfidf"][corpus], num_features=len(dct))
        self.answers = {
            l: a for l, a in set((lb, ans) for lb, ans in zip(etalons_df["label"], etalons_df["templateText"]))
        }

    async def classify(self, text: str, pub_id: int) -> SearchResponse:
        tokens = self.lemmatizer.tokenization([text])

        in_corpus = self.dictionary.doc2bow(tokens[0])
        in_vector = self.models["tfidf"][in_corpus]
        sims = self.index[in_vector]
        tfidf_tuples = [(num, scr) for num, scr in enumerate(list(sims), start=1) if scr >= self.params.score_threshold]
        if not tfidf_tuples:
            raise ScoreTooLow(f"scores are too low for text '''{text}'''")

        tfidf_best = sorted(tfidf_tuples, key=lambda x: x[1], reverse=True)[0]

        return SearchResponse(
            templateId=tfidf_best[0],
            templateText=self.answers[tfidf_best[0]],
            etalon_text="",
            algorithm="TFIDF",
            score=tfidf_best[1],
        )
