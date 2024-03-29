import json
import logging
import os
import socket
from datetime import datetime, date


# Level: Fatal/Error/Warn/Info/Debug/Trace
logLevels = {0: "Trace", 10: "Debug", 20: "Info", 30: "Warn", 40: "Error", 50: "Fatal"}


class CustomEncoder(json.JSONEncoder):
    """
    Serializes
    - date to iso string
    - exception to string
    """

    def default(self, o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, Exception):
            return str(o)

        return super().default(o)


def setup_logging(handlers, level=logging.INFO):
    """Sets up logging."""
    logging.root.setLevel(level)

    for handler in handlers:
        logging.root.addHandler(handler)


class JsonLogFormatter(logging.Formatter):
    """
    https://conf.action-media.ru/display/DEV/logging
    """

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord, *args, **kwargs):
        try:
            message = str(record.msg) % record.args
        except Exception:
            message = record.msg

        data = {
            "message": message,
            "application-name": "webapp",  # webapp/scheduleapp/desktopapp
            "level": logLevels.get(record.levelno, "Info"),
            "data": {
                "created": record.created,
                "filename": record.filename,
                "funcName": record.funcName,
                "module": record.module,
                "lineno": record.lineno,
                "name": record.name,
                "pathname": record.pathname,
                "process": record.process,
                "processName": record.processName,
                "thread": record.thread,
                "threadName": record.threadName,
            },
        }

        if "PRODUCT_NAME" in os.environ:
            data["product-name"] = os.environ.get("PRODUCT_NAME")

        if "SERVICE_NAME" in os.environ:
            data["service-name"] = os.environ.get("SERVICE_NAME")

        if "SERVICE_BRANCH_NAME_SANITIZED" in os.environ:
            data["service-branch-name"] = os.environ.get("SERVICE_BRANCH_NAME_SANITIZED")

        if "GIT_COMMIT_SHA" in os.environ:
            data["service-branch-sha"] = os.environ.get("GIT_COMMIT_SHA")

        if "SERVER_NAME" in os.environ:
            data["server-name"] = os.environ.get("SERVER_NAME")

        if "data" in record.__dict__:
            data["data"] = {**data["data"], **record.__dict__["data"]}

        if "additional-data" in record.__dict__:
            data["additional-data"] = record.__dict__["additional-data"]

        if "exception" in record.__dict__:
            data["exception"] = record.__dict__["exception"]

        return json.dumps(data, ensure_ascii=False, cls=CustomEncoder)


class UnixSocketHandler(logging.Handler):
    """Socket handler for logging to a Unix socket."""

    def __init__(self, address: str):
        logging.Handler.__init__(self)
        self.address = address
        self.formatter = JsonLogFormatter()
        self._connect_unixsocket(address)

    def _connect_unixsocket(self, address: str) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self.socket.connect(address)
        except socket.error:
            self.socket.close()
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(address)

    def close(self) -> None:
        self.socket.close()
        logging.Handler.close(self)

    def emit(self, record: logging.LogRecord):
        msg = self.format(record) + "\n"
        try:
            try:
                self.socket.send(msg.encode())
            except socket.error:
                self._connect_unixsocket(self.address)
                self.socket.send(msg.encode())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
