from .logging import logger, log_dist
from .merak_args import get_args
import numpy as np
import random

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)
