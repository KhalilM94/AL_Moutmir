import logging
import os
import sys
import tempfile
from typing import Optional

class TrainingLogger:
    """
    A utility class for configuring and retrieving a logger for training pipelines.
    Logs are written to both console and a timestamped log file.
    """

    def __init__(
        self,
        name: str = 'ML',
        log_dir: Optional[str] = None,
        log_filename: str = "logger",
        enable_file_logging: bool = True,
    ) -> None:
        """
        Initialize the logger with a given name and log directory.

        Args:
            name (str): Name of the logger. Default is 'ML'.
            log_dir (str): Directory where log files are written. Defaults to a private
                temporary directory owned by this logger.
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # mkdtemp, not TemporaryDirectory: a log file must outlive the logger object. The previous
        # default argument `os.path.join(tempfile.TemporaryDirectory().name, "logs")` was evaluated
        # once at import (so every logger shared one directory) and kept only `.name`, so the
        # discarded handle's finalizer deleted that directory out from under the open file
        # handlers - and main.py uploads that file to MLflow after training. mkdtemp has no
        # finalizer, so the logs survive until the OS reaps /tmp.
        self._owns_log_dir = log_dir is None
        if log_dir is None:
            log_dir = os.path.join(tempfile.mkdtemp(prefix="yg_eo_soilnet_logs_"), "logs")
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.enable_file_logging = enable_file_logging
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """
        Sets up file and stream handlers for the logger.

        Args:
            log_dir (str): Directory where log files will be saved.
            name (str): Name of the logger used in the filename.
        """
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.log_file: Optional[str] = None
        if self.enable_file_logging:
            self.log_file = os.path.join(self.log_dir, f"{self.log_filename}_training.log")
            os.makedirs(self.log_dir, exist_ok=True)

        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        if self.log_file is not None:
            file_handler: logging.FileHandler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger
