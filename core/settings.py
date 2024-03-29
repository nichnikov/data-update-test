import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from core.utils.logger import UnixSocketHandler, setup_logging


PROJECT_ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
# MODELS_DIR = os.path.join(DATA_DIR, "models")

# CONFIG_FILE = os.path.join(PROJECT_ROOT_DIR, "classifiers_config.yml")
MAPPING_FILE = os.path.join(DATA_DIR, "sys_pub_mappings.json")
ENV_FILE = os.path.join(PROJECT_ROOT_DIR, ".env")

print("PROJECT_ROOT_DIR:", PROJECT_ROOT_DIR)
print("DATA_DIR:", DATA_DIR)
print("ENV_FILE:", ENV_FILE)

class ProjectSettings(BaseSettings):
    """Project settings."""

    app_name: str = "expert_bot"
    host: str = "0.0.0.0"
    port: int = 8080

    log_level: int = logging.INFO
    logs_socket: str = "/var/log/sockets/type_python"


load_dotenv(dotenv_path=ENV_FILE)
project_settings = ProjectSettings()

handlers = [logging.StreamHandler()]
if os.path.exists(project_settings.logs_socket):
    handlers.append(UnixSocketHandler(project_settings.logs_socket))

setup_logging(handlers, project_settings.log_level)
