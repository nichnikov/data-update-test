import logging
from collections import namedtuple

from pydantic_settings import BaseSettings, SettingsConfigDict
from pymssql import connect

ROW = namedtuple(
    "ROW",
    "SysID, ID, Cluster, ParentModuleID, ParentID, ParentPubList, "
    "ChildBlockModuleID, ChildBlockID, ModuleID, Topic, Subtopic, DocName, ShortAnswerText",
)

logger = logging.getLogger(__name__)


class MSSQLSettings(BaseSettings):
    """MS SQL settings."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", env_prefix="ms_", extra="ignore")
    print("model_config:", model_config)

    host: str
    port: int
    user: str
    password: str
    database: str = "master"
    as_dict: bool = True
    charset: str = "cp1251"


class SQLDataFetcher:
    """Class for getting data from MS Server with specific SQL Query."""

    def __init__(self):
        self.ms_set = MSSQLSettings().model_dump()

    def establish_connection(self):
        """Establish connection to MS SQL Server."""
        try:
            conn = connect(**self.ms_set)
        except Exception as e:
            logger.error(e)
        return conn

    def fetch_from_db(self, sys_id: int, date: str):
        """
        Fetch data from the database for a given sys_id and date.

        :param sys_id: An integer representing the sys_id.
        :param date: A string representing the date.

        :return: A list of rows fetched from the database.
        """
        today_str = "'" + str(date) + "'"
        query = (
            "SELECT * FROM StatisticsRAW.[search].FastAnswer_RBD  "
            "WHERE SysID = {} AND (ParentBegDate <= {} AND ParentEndDate IS NULL "
            "OR ParentBegDate <= {} AND  ParentEndDate >= {})".format(sys_id, today_str, today_str, today_str)
        )

        conn = self.establish_connection()
        with conn.cursor() as cursor:
            cursor.execute(query)
            data_from_db = cursor.fetchall()

        return data_from_db

    def get_rows(self, sys_id: int, date: str) -> list:
        """
        Parsing rows from DB and returning list of unique tuples with etalons and list of tuples with data for answers
        """
        rows = []
        data_from_db = self.fetch_from_db(sys_id, date)
        for row in data_from_db:
            try:
                parent_pub_list = [int(pb) for pb in row["ParentPubList"].split(",") if pb != ""]
                rows.append(
                    ROW(
                        row["SysID"],
                        row["ID"],
                        row["Cluster"],
                        row["ParentModuleID"],
                        row["ParentID"],
                        parent_pub_list,
                        row["ChildBlockModuleID"],
                        row["ChildBlockID"],
                        row["ModuleID"],
                        row["Topic"],
                        row["Subtopic"],
                        row["DocName"],
                        row["ShortAnswerText"],
                    )
                )

            except ValueError as err:
                logger.exception("Parsing %s with row: %s", err, row)
        logger.info("Unique etalons tuples rows quantity is %s for SysID %s", len(rows), sys_id)
        return rows
