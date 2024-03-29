import logging
import os

import boto3
from botocore.exceptions import ClientError
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class S3Settings(BaseSettings):
    """Settings for S3 storage"""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", env_prefix="s3_", extra="ignore")

    bucket_name: str = "nlp-models"
    endpoint_url: str = "https://s3.dev.aservices.tech:443"

    user: str
    access_key: str
    secret_key: str


class S3ModelDownloader:
    """Client to operate with particular bucket in S3 storage"""

    def __init__(self):
        self.s_3 = S3Settings()
        self.resource = boto3.resource(
            "s3",
            aws_access_key_id=self.s_3.access_key,
            aws_secret_access_key=self.s_3.secret_key,
            endpoint_url=self.s_3.endpoint_url,
        )

    def save_model(self, model_name: str) -> None:
        """
        Downloads model from S3 storage and saves it locally.

        :param model_name: folder name in S3 storage and local storage.
        """
        if not model_name:
            logger.warning("Model name is not provided.")
            return

        try:
            bucket = self.resource.Bucket(self.s_3.bucket_name)
            paginator = self.resource.meta.client.get_paginator("list_objects")

            for result in paginator.paginate(Bucket=self.s_3.bucket_name, Delimiter="/", Prefix=model_name):
                self._process_common_prefixes(result.get("CommonPrefixes", []))
                self._process_contents(bucket, result.get("Contents", []))

        except Exception as error:
            logger.error("Error saving model %s: %s", model_name, error)

    def _process_common_prefixes(self, common_prefixes):
        for subdir in common_prefixes:
            self.save_model(subdir.get("Prefix"))

    def _process_contents(self, bucket, contents):
        for file in contents:
            dest_pathname = os.path.join(MODELS_DIR, file.get("Key"))
            self._create_directory_if_needed(dest_pathname)
            self._download_file_if_needed(bucket, file.get("Key"), dest_pathname)

    @staticmethod
    def _create_directory_if_needed(path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            logger.info("Creating folder %s", directory)
            os.makedirs(directory)

    @staticmethod
    def _download_file_if_needed(bucket, key, dest_pathname):
        if not os.path.exists(dest_pathname):
            logger.info("Downloading file %s", dest_pathname)
            try:
                bucket.download_file(key, dest_pathname)
            except ClientError as error:
                logger.error("Error downloading file %s: %s", dest_pathname, error)
