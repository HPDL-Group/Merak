from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist


class Metric(object):
    def __init__(self, name: str):
        self.name = name
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor):
        self.sum += val
        self.n += 1

    def reset(self):
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    @property
    def avg(self) -> torch.Tensor:
        return self.sum / self.n


class AccMetric(object):
    def __init__(self):
        self.metrics = {}

    def update(self, key: str, val: torch.Tensor):
        if key in self.metrics:
            self.metrics[key].update(val)
        else:
            self.metrics[key] = Metric(key)
            self.metrics[key].update(val)

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

    @property
    def avg(self) -> Dict[str, torch.Tensor]:
        avg_dict = {}
        for key in self.metrics:
            avg_dict[key] = self.metrics[key].avg.item()
        self.reset()

        return avg_dict
