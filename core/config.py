import os

import pandas as pd
import yaml
from pydantic import field_validator, BaseModel, Extra
from pydantic_settings import BaseSettings

from core.classifiers import classifier_classes
from core.exceptions import ConfigError
from core.settings import CONFIG_FILE, MAPPING_FILE, DATA_DIR
from core.utils.other import read_json


class ClassifierParams(BaseModel, extra=Extra.allow):
    """Common params used by all classifiers."""

    class_name: str
    score_threshold: float | None = None
    len_limit: int | None = None
    stopwords_files: list[str] | None = None
    num_candidates: int | None = None
    es_clusters_index: str | None = None
    es_answers_index: str | None = None

    @field_validator("stopwords_files")
    def validate_files(cls, value):
        """Check if stopwords files exist"""
        for file_name in value:
            file_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(file_path):
                raise ConfigError(f"Stopwords file {file_name} doesn't exist")
        return value

    @property
    def stopwords(self) -> list[str]:
        """Read stopwords from files"""
        stopwords = []
        for file_name in self.stopwords_files:
            file_path = os.path.join(DATA_DIR, file_name)
            stopwords_df = pd.read_csv(str(file_path), sep="\t")
            stopwords.extend(stopwords_df["stopwords"].tolist())
        return stopwords

    @field_validator("class_name")
    def validate_class_name(cls, value):
        """Check if classifier class is declared in classifiers module"""

        if value not in classifier_classes:
            raise ConfigError(f"Unknown classifier class in classifiers_conf: {value}")
        return value


class ClassifiersConfig(BaseSettings):
    """Classifiers classifiers_conf."""

    params: dict[str, ClassifierParams] = None
    scenarios: dict[str, list[str]] = None
    pub_sys_mapping: dict = {v: int(k) for k, l in read_json(MAPPING_FILE).items() for v in l}

    @field_validator("params", mode="before")
    def read_params(cls, v):
        """Read classifier params from yaml file"""

        with open(CONFIG_FILE, "r", encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)["classifiers"]

        for classifier_name, classifier_params in settings.items():
            settings[classifier_name] = ClassifierParams(**classifier_params)

        return v or settings

    @field_validator("scenarios", mode="before")
    def read_scenarios(cls, v):
        """Read work scenarios from yaml file"""

        with open(CONFIG_FILE, "r", encoding="utf-8") as stream:
            _config = yaml.safe_load(stream)
            declared_classifiers = set(_config["classifiers"].keys())

            scenarios_section = _config["scenarios"]
            sequence_to_prepend = scenarios_section.pop("prepend_to_all", [])

            default_scenario = [*sequence_to_prepend, *scenarios_section.pop("default")]
            scenarios_by_sys_id = scenarios_section.pop("by_sys_id", {})

        if not set(default_scenario).issubset(declared_classifiers):
            raise ConfigError("Unknown classifier in default scenario")

        for sys_id, classifiers in scenarios_by_sys_id.items():
            if not set(classifiers).issubset(declared_classifiers):
                raise ConfigError(f"Unknown classifier in scenario for sys_id {sys_id}")

        scenarios = {
            str(sys_id): [*sequence_to_prepend, *classifiers] for sys_id, classifiers in scenarios_by_sys_id.items()
        }
        scenarios["default"] = default_scenario
        return v or scenarios


class UpdateServiceConfig(BaseModel):
    """Config for update service."""

    sys_files_pubs: dict[str, dict[str, list[int]]] = None
    appendix: int = 1000000
    LemCluster: bool = True
    LemDocName: bool = True
    LemShortAnswerText: bool = True
    stopwords_files: list[str] = None

    @field_validator("stopwords_files", mode="before")
    def validate_files(cls, value):
        """Check if stopwords files exist"""
        for file_name in value:
            file_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(file_path):
                raise ConfigError(f"Stopwords file {file_name} doesn't exist")
        return value

    @property
    def stopwords(self) -> list[str]:
        """Read stopwords from files"""
        stopwords = []
        for file_name in self.stopwords_files:
            file_path = os.path.join(DATA_DIR, file_name)
            stopwords_df = pd.read_csv(file_path, sep="\t")
            stopwords.extend(stopwords_df["stopwords"].tolist())
        return stopwords


classifiers_conf = ClassifiersConfig()
update_config = UpdateServiceConfig()
