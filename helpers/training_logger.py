import logging
import os
import sys
import tempfile

class TrainingLogger:
    """
    A utility class for configuring and retrieving a logger for training pipelines.
    Logs are written to both console and a timestamped log file.
    """

    def __init__(self, name: str = 'ML', 
                 log_dir: str = os.path.join(tempfile.TemporaryDirectory().name, "logs"), 
                 log_filename: str = "logger") -> None:
        """
        Initialize the logger with a given name and log directory.

        Args:
            name (str): Name of the logger. Default is 'ML'.
            log_dir (str): Directory where log files will be saved. Default is 'logs'.
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_dir = log_dir 
        self.log_filename = log_filename
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

        self.log_file: str = os.path.join(self.log_dir, f"{self.log_filename}_training.log")
        os.makedirs(self.log_dir, exist_ok=True)

        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        file_handler: logging.FileHandler = logging.FileHandler(self.log_file)
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
