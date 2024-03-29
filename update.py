import json
import logging
import os
from collections import namedtuple
from datetime import datetime

import pandas as pd
from pymystem3 import Mystem

from core.elastic.client import ElasticClient
from core.elastic.queries import Match
from core.mssql import SQLDataFetcher
from core.settings import DATA_DIR
from core.text_preprocessing.lemmatizer import TextLemmatizer
from core.utils.other import chunks

logger = logging.getLogger(__name__)


class UpdateService:
    """
    Responsible for updating data in Elasticsearch
    indexes based on different sources such as MS statistics and CSV files.
    """

    def __init__(
        self,
        es_client: ElasticClient,
        db_conn: SQLDataFetcher,
        mystem: Mystem,
        # first_sents_extraction: FirstSentenceExtractor,
    ):
        self.es_client = es_client
        self.db_conn = db_conn
        self.mystem = mystem

        # self.first_sents_extraction = first_sents_extraction

    def texts_tokenize(self, texts: list[str], stopwords_roots: list[str]):
        tokenizer = TextLemmatizer(mystem=self.mystem)
        stopwords = []

        for root in stopwords_roots:
            stopwords_df = pd.read_csv(root, sep="\t")
            stopwords += list(stopwords_df["stopwords"])
        tokenizer.add_stopwords(stopwords)

        results = []
        for texts_chunk in chunks(texts, 15000):
            results += [" ".join(lm_text) for lm_text in tokenizer.tokenization(texts_chunk)]

        return results

    def update_data_with_lemmas(self, data_dicts: list[dict], **kwargs):
        """Update the given data dictionaries with lemmas."""

        sws_roots = []
        if kwargs["stopwords_files"]:
            for file_name in kwargs["stopwords_files"]:
                sws_roots.append(os.path.join(DATA_DIR, file_name))

        clusters = [str(x["Cluster"]) for x in data_dicts]
        lem_clusters = self.texts_tokenize(clusters, sws_roots)
        for _dict, l_c in zip(data_dicts, lem_clusters):
            _dict["LemCluster"] = l_c

        if kwargs["LemDocName"]:
            doc_names = [str(x["DocName"]) for x in data_dicts]
            lem_doc_names = self.texts_tokenize(doc_names, sws_roots)
            for _dict, l_dn in zip(data_dicts, lem_doc_names):
                _dict["LemDocName"] = l_dn

        if kwargs["LemShortAnswerText"]:
            short_answers = [str(x["ShortAnswerText"]) for x in data_dicts]
            lem_short_answers = self.texts_tokenize(short_answers, sws_roots)
            for _dict, l_sa in zip(data_dicts, lem_clusters, lem_short_answers):
                _dict["LemShortAnswerText"] = l_sa

    async def get_msdb_data(self, **kwargs):
        ROW_FOR_ANSWERS = namedtuple(
            "ROW_FOR_ANSWERS",
            "SysID, ID, ParentModuleID, ParentID, ChildBlockModuleID, ChildBlockID, ShortAnswerText",
        )

        def data_for_answer_create(pubs_urls: list, row_tuples: list[ROW_FOR_ANSWERS]):
            pubs_answers = []
            for pub, sys_url in pubs_urls:
                for row_tuple in row_tuples:
                    if row_tuple.ParentModuleID == 16 and row_tuple.ChildBlockModuleID in [86, 12]:
                        module_id = row_tuple.ChildBlockModuleID
                        document_id = row_tuple.ChildBlockID
                    else:
                        module_id = row_tuple.ParentModuleID
                        document_id = row_tuple.ParentID

                    query_url = "/".join([sys_url, str(module_id), str(document_id), "actual/"])

                    """
                    # Выключение добавления первого предложения:
                    first_sentence = self.first_sents_extraction([row_tuple.ShortAnswerText])
                    first_sentence = [s for s in first_sentence if s is not None]
                    if first_sentence:
                        variants = [
                            "Далее см.",
                            "Подробнее см.",
                            "Читайте подробнее",
                            "Ссылка по вашему вопросу",
                            "Смотрите подробнее",
                            "Подробнее смотрите",
                            "Подробнее в материале",
                            "Вот ссылка по вашему вопросу",
                            "Далее читайте",
                        ]
                        answer_text = " ".join([first_sentence[0], choice(variants)])
                    else:
                        answer_text = "Вот ссылка по вашему вопросу: "
                    """

                    answer_text = "Вот материал по вашему вопросу. Если это не совсем то, что нужно, я продолжу поиск "
                    pubs_answers.append(
                        {
                            "pubId": int(pub),
                            "templateId": int(row_tuple.ID),
                            "templateText": " ".join([answer_text, str(query_url)]),
                        }
                    )
            return pubs_answers

        today = datetime.today().strftime("%Y-%m-%d")
        result_clusters, result_answers = [], []
        for sys_id in kwargs["sys_pub_url"]:
            rows = self.db_conn.get_rows(int(sys_id), today)
            data_dicts = [nt._asdict() for nt in rows]
            
            # добавление ParentPubListSys и присвоение этому элементу значений из ParentPubList
            for d in data_dicts:
                d["ParentPubListSys"] = d.pop("ParentPubList")
                
            # добавление ParentPubList с PubIds из statistics_parameters
            pubs = [x[0] for x in kwargs["sys_pub_url"][sys_id]]
            for d in data_dicts:
                d["ParentPubList"] = pubs
            
            self.update_data_with_lemmas(data_dicts, **kwargs)
            rows_answers = [
                ROW_FOR_ANSWERS(
                    r.SysID, r.ID, r.ParentModuleID, r.ParentID, r.ChildBlockModuleID, r.ChildBlockID, r.ShortAnswerText
                )
                for r in rows
            ]

            rows_answers_unique = list(set(rows_answers))

            result_clusters.extend(data_dicts)
            result_answers.extend(data_for_answer_create(kwargs["sys_pub_url"][sys_id], rows_answers_unique))
        return result_clusters, result_answers

    async def scv2es(self, **kwargs):
        """Обновление данных в индексе "clusters" из csv файлов"""
        ANS = namedtuple("ANS", "pubId, templateId, templateText")
        for _key, value in kwargs.items():
            for sys_id in value["sys_files_pubs"]:
                appendix = value["appendix"] * int(sys_id)
                file_name = value["sys_files_pubs"][sys_id]["file_name"]
                pubs = value["sys_files_pubs"][sys_id]["pubs"]
                dataframe = pd.read_csv(str(os.path.join(DATA_DIR, file_name)), sep="\t")
                data_dicts = dataframe.to_dict(orient="records")

                clusters_for_es = [
                    {
                        "SysID": int(sys_id),
                        "ID": int(appendix) + int(d["templateId"]),
                        "Cluster": d["text"],
                        "ParentModuleID": 0,
                        "ParentID": 0,
                        "ParentPubList": pubs,
                        "ChildBlockModuleID": 0,
                        "ChildBlockID": 0,
                        "ModuleID": 85,
                        "Topic": "нет",
                        "Subtopic": "нет",
                        "DocName": "нет",
                        "ShortAnswerText": d["templateText"],
                    }
                    for d in data_dicts
                ]

                answers = [ANS(pubid, d["ID"], d["ShortAnswerText"]) for d in clusters_for_es for pubid in pubs]
                answers_for_es = [x._asdict() for x in set(answers)]

                self.update_data_with_lemmas(clusters_for_es, **value)

                # добавление вопросов и ответов:
                await self.es_client.add_docs(value["clusters_index"], clusters_for_es)
                await self.es_client.add_docs(value["answers_index"], answers_for_es)

    async def run(self):
        """
        Runs the update service.

        Example usage:
            update_service = UpdateService()
            await update_service.run()
        """
        with open(os.path.join(DATA_DIR, "csv_parameters.json"), "r", encoding="utf-8") as st_f:
            csv_prmtrs = json.load(st_f)

        with open(os.path.join(DATA_DIR, "statistics_parameters.json"), "r", encoding="utf-8") as st_f:
            stat_prmtrs = json.load(st_f)

        indexes = [
            stat_prmtrs["clusters_index_name"],
            stat_prmtrs["answers_index_name"],
            stat_prmtrs["greetings_index_name"],
        ]

        msdb_clusters, msdb_answers = await self.get_msdb_data(**stat_prmtrs)
        if not msdb_clusters or not msdb_answers:
            logger.info("Данные для обновления не найдены в msdb. Завершение работы")
            return

        logger.info("0. Удаление устаревших данных")
        for index in indexes:
            await self.es_client.delete_index(index)
            await self.es_client.create_index(index)

        logger.info("1. Добавление эталонов и ответов")
        await self.es_client.add_docs(stat_prmtrs["clusters_index_name"], msdb_clusters)
        await self.es_client.add_docs(stat_prmtrs["answers_index_name"], msdb_answers)

        logger.info("2. Добавление из csv файлов")
        await self.scv2es(**csv_prmtrs)

        logger.info("3. Удаление эталонов и ответов по списку")
        dataframe = pd.read_csv(os.path.join(DATA_DIR, "del_answers.csv"), sep="\t")
        for template_id in dataframe["TemplateId"]:
            await self.es_client.q_delete("clusters", Match("ID", template_id))
            await self.es_client.q_delete("answers", Match("templateId", template_id))

        await self.es_client.close()


if __name__ == "__main__":
    import asyncio

    es = ElasticClient()
    db_con = SQLDataFetcher()
    mystem  = Mystem()

    srv = UpdateService(es, db_con, mystem)
    asyncio.run(srv.run())
    pass