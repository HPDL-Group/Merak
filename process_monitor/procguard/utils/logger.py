# coding=utf-8
# Copyright (c) 2022, HPDL group, NUDT, PDL lab.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logging Utilities - Helper functions for configuring Python logging.

This module provides convenient functions for setting up logging configuration
for ProcGuard and other applications. It supports both console and file logging
with customizable format strings.

Functions:
- setup_logging: Configure root logger with handlers
- get_logger: Get a logger instance by name

Example:
    >>> from procguard.utils.logger import setup_logging, get_logger
    >>> setup_logging(level="DEBUG", log_file="app.log")
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    """
    Configure the root logger with console and optionally file handlers.

    This function sets up logging for the entire application by configuring
    the root logger. It creates a console handler and optionally a file handler
    with the specified log level and format.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for file logging
        log_format: Format string for log messages

    Example:
        >>> setup_logging(level="DEBUG", log_file="procguard.log")
        >>> # All loggers will now output to console and file
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name for the logger (usually __name__)

    Returns:
        logging.Logger: Logger instance with the given name

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return logging.getLogger(name)
