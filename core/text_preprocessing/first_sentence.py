import re

from nltk import sent_tokenize


# Requires nltk punkt in /usr/local/share/nltk_data/tokenizers/


class FirstSentenceExtractor:
    """Extracts the first sentence from a text."""

    def __init__(self, patterns):
        self.pttns = patterns

    @staticmethod
    def _one_text_splited(sentences: list[str]) -> str:
        """объединение первых предложений в один текст"""
        if not sentences:
            return ""

        first_sen = sentences[0]
        if len(first_sen.split()) <= 2:
            for sen in sentences[1:]:
                if len(sen.split()) <= 3:
                    first_sen += " " + sen
                else:
                    return first_sen + " " + sen
        else:
            return first_sen

    def patterns_changes(self, text: str):
        """Replace patterns in a given text."""
        for asc, dsc in self.pttns:
            text = re.sub(asc, dsc, text, flags=re.IGNORECASE)
        return text

    def first_sentence_extraction(self, texts: list[str]) -> list:
        """Extract the first sentence from a list of texts."""
        working_texts = [self.patterns_changes(tx) for tx in texts]

        sents_tokens = [sent_tokenize(tx, language="russian") for tx in working_texts]
        results = []
        for sens in sents_tokens:
            sens = [s for s in sens if s is not None]
            if sens:
                sentence = self._one_text_splited(sens)
                results.append(sentence)
        return results

    def __call__(self, texts):
        return self.first_sentence_extraction(texts)
