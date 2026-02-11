# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Yck (eyichenke@gmail.com)
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

from typing import Any, Callable, Dict, List, Optional, Tuple


def _snake_case(s: str):
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append("_")
        chars.append(c.lower())
        prev_lower = c.islower()
    return "".join(chars)


def _get_count(param_count: Dict[str, int], node_name: str):
    """Identify different mutations of a given node name."""

    if node_name in param_count:
        return param_count[node_name]
    elif node_name.replace(".", "_") in param_count:
        return param_count[node_name.replace(".", "_")]
    else:
        raise RuntimeError(
            f"Unable to find match between \
                             param {param_count} and node {node_name}"
        )


def _create_shard_to_param_count(
    param_count: Dict[str, int], node_name_to_shard_id: Dict[str, int]
) -> Dict[str, int]:
    """Utility to create a map from shard id to param count using
    existing state."""

    shard_to_param_count: Dict[int, int] = {}
    for node in node_name_to_shard_id.keys():
        try:
            count = _get_count(param_count, node)
        except RuntimeError:
            # print_rank_0(f"Unable to find match node {node_name}")
            continue
        if node_name_to_shard_id[node] in shard_to_param_count:
            shard_to_param_count[node_name_to_shard_id[node]] += count
        else:
            shard_to_param_count[node_name_to_shard_id[node]] = count
    return shard_to_param_count
