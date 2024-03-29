import logging
import operator
import re
from itertools import groupby

from pymystem3 import Mystem

logger = logging.getLogger(__name__)


class TextLemmatizer:
    def __init__(self, mystem: Mystem):
        self._stopwords = []
        self._synonyms = []
        self.stopwords_patterns = re.compile("")

        self.mystem = mystem

    @staticmethod
    def _group_by_value(asc_dsc: list):
        sorted_items = sorted(asc_dsc, key=lambda x: x[1])
        grouped_items = groupby(sorted_items, operator.itemgetter(1))
        for key, value in grouped_items:
            yield key, [x[0] for x in value]

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return re.sub(r"[^\w\n\s]", " ", text)

    def lemmatize_text(self, text: str) -> str:
        """Lemmatization for text. It returns lemmatized text"""

        text_ = self._preprocess_text(text)
        lm_text = "".join(self.mystem.lemmatize(text_.lower())).strip()

        return lm_text

    def lemmatize_texts(self, texts: list[str]) -> list[list[str]]:
        """Lemmatization for texts in list. It returns list with lemmatized texts"""

        text_ = self._preprocess_text("\n".join(texts))
        lm_texts = "".join(self.mystem.lemmatize(text_.lower()))
        return [lm_tx.split() for lm_tx in lm_texts.split("\n")][:-1]

    def add_stopwords(self, stopwords: list[str]):
        """adding stop words into class"""

        self._stopwords = [" ".join(x) for x in self.lemmatize_texts(stopwords)]
        self.stopwords_patterns = re.compile("|".join([r"\b" + tx + r"\b" for tx in self._stopwords]))

    def add_synonyms(self, synonyms: list[str]):
        """adding stop words into class"""

        ascs, dscs = zip(*synonyms)
        lm_ascs = self.tokenization(list(ascs))
        syns_dct = dict(self._group_by_value(list(zip(lm_ascs, dscs))))
        for asc in syns_dct:
            self._synonyms.append((asc, re.compile("|".join([r"\b" + w + r"\b" for w in syns_dct[asc]]))))

    def tokenization(self, texts: list[str]) -> list[list[str]]:
        """list of texts lemmatization with stop words deleting"""

        lemm_texts = self.lemmatize_texts(texts)
        if self._synonyms:
            lem_texts_union = "\n".join([" ".join(lm_tx) for lm_tx in lemm_texts])
            for syn_pair in self._synonyms:
                lem_texts_union = syn_pair[1].sub(syn_pair[0], lem_texts_union)
            if self._stopwords:
                return [self.stopwords_patterns.sub(" ", l_tx).split() for l_tx in lem_texts_union.split("\n")]
            return [l_tx.split() for l_tx in lem_texts_union.split("\n")]

        if self._stopwords:
            return [self.stopwords_patterns.sub(" ", " ".join(l_tx)).split() for l_tx in lemm_texts]

        return lemm_texts
