from .jaccard_classifier import JaccardClassifier
from .kosgu_classifier import KosguClassifier
from .sbert_classifier import SBERTClassifier
from .sbert_t5_classifier import SBERTT5Classifier
from .tfidf_classifier import TFIDFClassifier

classifier_classes = {
    cls.__name__: cls
    for cls in [SBERTClassifier, TFIDFClassifier, JaccardClassifier, KosguClassifier, SBERTT5Classifier]
}
