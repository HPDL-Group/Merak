# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
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
ProcGuard Configuration Package

This package provides configuration management for ProcGuard including:
- Schema definitions for all configuration types
- YAML configuration file loading and parsing
- Configuration validation

Modules:
- schema: Dataclass definitions for all configuration types
- loader: YAML configuration file loader and parser

Classes:
- ConfigLoader: Loads and parses YAML configuration files
- WorkerConfig: Worker process configuration
- ProcGuardConfig: Root configuration container
- SLURMConfig: SLURM cluster integration settings
- PyTorchDistConfig: PyTorch distributed training settings
"""

from .loader import ConfigLoader
from .schema import WorkerConfig, ProcGuardConfig, SLURMConfig, PyTorchDistConfig

__all__ = ["ConfigLoader", "WorkerConfig", "ProcGuardConfig", "SLURMConfig", "PyTorchDistConfig"]
