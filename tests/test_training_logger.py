"""TrainingLogger owns its temp directory.

log_dir used to default to `os.path.join(tempfile.TemporaryDirectory().name, "logs")`, which was
evaluated once at import (shared across every logger) and kept only `.name`, so the discarded
handle's finalizer deleted the directory while file handlers still pointed at it.
"""

import gc
import os

from yg_eo_soilnet.logger.training_logger import TrainingLogger


def test_each_logger_gets_its_own_directory() -> None:
    first = TrainingLogger(name="logger-a", log_filename="a")
    second = TrainingLogger(name="logger-b", log_filename="b")

    assert first.log_dir != second.log_dir


def test_log_file_survives_garbage_collection() -> None:
    """The old default let a finalizer delete the directory out from under the handlers."""
    logger = TrainingLogger(name="logger-gc", log_filename="gc")
    logger.get_logger().info("written before collection")

    gc.collect()

    assert logger.log_file is not None
    assert os.path.exists(logger.log_file), "log file was removed by a finalizer"


def test_default_log_dir_has_no_finalizer_that_could_delete_it() -> None:
    """TemporaryDirectory would warn and delete on GC; a log directory must simply persist."""
    logger = TrainingLogger(name="logger-handle", log_filename="handle")

    assert logger._owns_log_dir is True
    assert "yg_eo_soilnet_logs_" in logger.log_dir
    assert not hasattr(logger, "_temp_dir")


def test_explicit_log_dir_is_honoured_without_creating_a_temp_dir(tmp_path) -> None:
    logger = TrainingLogger(name="logger-explicit", log_dir=str(tmp_path), log_filename="explicit")

    assert logger._owns_log_dir is False
    assert logger.log_dir == str(tmp_path)
    assert os.path.exists(logger.log_file)


def test_file_logging_can_be_disabled() -> None:
    logger = TrainingLogger(name="logger-off", log_filename="off", enable_file_logging=False)

    assert logger.log_file is None
