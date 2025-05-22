import logging
import os
import sys
from datetime import datetime

class TrainingLogger:
    """
    A utility class for configuring and retrieving a logger for training pipelines.
    Logs are written to both console and a timestamped log file.
    """

    def __init__(self, name: str = 'ML', log_dir: str = 'logs') -> None:
        """
        Initialize the logger with a given name and log directory.

        Args:
            name (str): Name of the logger. Default is 'ML'.
            log_dir (str): Directory where log files will be saved. Default is 'logs'.
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._setup_handlers(log_dir, name)

    def _setup_handlers(self, log_dir: str, name: str) -> None:
        """
        Sets up file and stream handlers for the logger.

        Args:
            log_dir (str): Directory where log files will be saved.
            name (str): Name of the logger used in the filename.
        """
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file: str = os.path.join(log_dir, f"{name}_training_{timestamp}.log")
        os.makedirs(log_dir, exist_ok=True)

        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger
