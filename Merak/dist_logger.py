import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch.distributed as dist


class CustomFormatter(logging.Formatter):
    """Custom log formatter that adapts format based on logger name.

    This formatter dynamically changes the log format based on the logger name,
    supporting different verbosity levels for different logging scenarios.
    """

    # Pre-defined format templates for different logging scenarios
    _FORMAT_TEMPLATES = {
        "simple": "%(message)s",  # Minimal output, just the message
        "normal": "[%(asctime)s] [%(levelname)s] %(message)s",  # Standard format with timestamp and level
        "detailed": "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",  # Full debug info
    }

    def __init__(self):
        """Initialize the custom formatter with default settings."""
        super().__init__()
        self._default_format = self._FORMAT_TEMPLATES["normal"]

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified log record based on logger name.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message string.

        Raises:
            ValueError: If logger name doesn't match any known format type.
        """
        # Determine format based on logger name
        logger_name = record.name

        if logger_name in self._FORMAT_TEMPLATES:
            self._style._fmt = self._FORMAT_TEMPLATES[logger_name]
        else:
            # For unknown logger names, use a reasonable default
            self._style._fmt = self._default_format

        return super().format(record)


class DistributedLogger(logging.Logger):
    """A distributed-aware logger that supports rank-based log filtering.

    This logger extends the standard logging.Logger to provide distributed
    training compatibility, allowing logs to be filtered by process rank.

    Attributes:
        name: Logger identifier.
        rank: Current process rank in distributed setting, -1 if not distributed.
        is_dist_initialized: Whether distributed training is initialized.
    """

    # Standard log level mappings
    _LOG_LEVEL_MAP: Dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, name: str):
        """Initialize the distributed logger.

        Args:
            name: Logger name, used to determine log format. Must be one of:
                  "simple", "normal", "detailed", or other custom names.

        Raises:
            ValueError: If name is None or empty.
        """
        if not name:
            raise ValueError("Logger name must be specified and non-empty")

        super().__init__(name)
        self.name = name

        # Initialize distributed training attributes
        self.rank: int = -1
        self.is_dist_initialized: bool = False
        self._update_distributed_status()

        # Configure logger level and handlers
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure logger level, handlers, and formatters."""
        # Set log level from environment variable or default to INFO
        env_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        self.level = self._resolve_log_level(env_level)

        # Create and configure console handler
        console_handler = self._create_console_handler()
        self.addHandler(console_handler)

        # Prevent propagation to parent loggers to avoid duplicate logs
        self.propagate = False

    def _create_console_handler(self) -> logging.Handler:
        """Create and configure a console handler for stdout.

        Returns:
            Configured console handler with custom formatter.
        """
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(self.level)
        handler.setFormatter(CustomFormatter())
        return handler

    def _resolve_log_level(self, level_str: str) -> int:
        """Resolve log level string to logging constant.

        Args:
            level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Returns:
            Corresponding logging level constant.

        Note:
            Falls back to INFO if the provided level string is invalid.
        """
        return self._LOG_LEVEL_MAP.get(level_str, logging.INFO)

    def _update_distributed_status(self) -> None:
        """Update distributed training status and current rank."""
        self.is_dist_initialized = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist_initialized else -1

    def _should_log(self, target_ranks: Optional[List[int]] = None) -> bool:
        """Determine whether current process should log based on target ranks.

        Args:
            target_ranks: List of ranks that should log. If None, all ranks log.

        Returns:
            True if current process should log, False otherwise.
        """
        # Always log if distributed training is not initialized
        if not self.is_dist_initialized:
            return True

        # Log if no specific ranks are targeted (all ranks)
        if target_ranks is None:
            return True

        # Log only if current rank is in target list
        return self.rank in set(target_ranks)

    def _log_with_caller_info(self, level: int, msg: str, ranks: Optional[List[int]] = None) -> None:
        """Log a message with specified level and rank filter.

        Args:
            level: Logging level constant.
            msg: Message to log.
            ranks: Specific ranks that should log this message.
                   If None, all ranks will log.
        """
        # Ensure distributed status is up to date
        self._update_distributed_status()

        if self._should_log(ranks):
            # Format and log the message using parent class implementation
            formatted_msg = f"{msg}"  # Placeholder for any message formatting
            super().log(level, formatted_msg, stacklevel=3)

    def info(self, msg: str, ranks: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Log an info level message.

        Args:
            msg: Informational message to log.
            ranks: Specific ranks that should log this message.
        """
        self._log_with_caller_info(logging.INFO, msg, ranks)

    def warning(self, msg: str, ranks: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Log a warning level message.

        Args:
            msg: Warning message to log.
            ranks: Specific ranks that should log this message.
        """
        self._log_with_caller_info(logging.WARNING, msg, ranks)

    def error(self, msg: str, ranks: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Log an error level message.

        Args:
            msg: Error message to log.
            ranks: Specific ranks that should log this message.
        """
        self._log_with_caller_info(logging.ERROR, msg, ranks)

    def debug(self, msg: str, ranks: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Log a debug level message.

        Args:
            msg: Debug message to log.
            ranks: Specific ranks that should log this message.
        """
        self._log_with_caller_info(logging.DEBUG, msg, ranks)

    def critical(self, msg: str, ranks: Optional[List[int]] = None, *args, **kwargs) -> None:
        """Log a critical level message.

        Args:
            msg: Critical message to log.
            ranks: Specific ranks that should log this message.
        """
        self._log_with_caller_info(logging.CRITICAL, msg, ranks)


# Convenience function for quick logger access
def get_distributed_logger(name: str = "normal") -> DistributedLogger:
    """Get a distributed logger instance with specified name.

    Args:
        name: Logger name, determines log format style.

    Returns:
        Configured DistributedLogger instance.
    """
    return DistributedLogger(name)
