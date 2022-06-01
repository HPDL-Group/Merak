"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
# https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/config.py

import os
from typing import Union

import torch
import json
import copy

from .config_utils import get_scalar_param, dict_raise_error_on_duplicate_keys, ScientificNotationEncoder, DeepSpeedConfigObject
from ..utils import logger

TENSOR_CORE_ALIGN_SIZE = 8

#############################################
# Routes
#############################################
ROUTE_TRAIN = "train"
ROUTE_EVAL = "eval"
ROUTE_PREDICT = "predict"
ROUTE_ENCODE = "encode"

#############################################
# Batch size
#############################################
TRAIN_BATCH_SIZE = "train_batch_size"
TRAIN_BATCH_SIZE_DEFAULT = None

# Steps
STEPS_PER_PRINT = "steps_per_print"
STEPS_PER_PRINT_DEFAULT = 10

#########################################
# Training micro batch size per GPU
#########################################
# Batch size for one training step. This is used when the
# TRAIN_BATCH_SIZE cannot fit in GPU memory to determine
# the number of gradient accumulation steps. By default, this
# is set to None. Users can configure in ds_config.json as below example:
TRAIN_MICRO_BATCH_SIZE_PER_GPU = '''
TRAIN_MICRO_BATCH_SIZE_PER_GPU is defined in this format:
"train_micro_batch_size_per_gpu": 1
'''
TRAIN_MICRO_BATCH_SIZE_PER_GPU = "train_micro_batch_size_per_gpu"
TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = None

#########################################
# Gradient Accumulation
#########################################
# Gradient accumulation feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_ACCUMULATION_FORMAT = '''
Gradient Accumulation should be of the format:
"gradient_accumulation_steps": 1
'''
GRADIENT_ACCUMULATION_STEPS = "gradient_accumulation_steps"
GRADIENT_ACCUMULATION_STEPS_DEFAULT = None

#########################################
# Gradient clipping
#########################################
# Gradient clipping. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_CLIPPING_FORMAT = '''
Gradient clipping should be enabled as:
"gradient_clipping": 1.0
'''
GRADIENT_CLIPPING = 'gradient_clipping'
GRADIENT_CLIPPING_DEFAULT = 0.

#########################################
# Scale/predivide gradients before allreduce
#########################################
# Prescale gradients. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
PRESCALE_GRADIENTS_FORMAT = '''
Gradient prescaling should be enabled as:
"prescale_gradients": true
'''
PRESCALE_GRADIENTS = "prescale_gradients"
PRESCALE_GRADIENTS_DEFAULT = False

GRADIENT_PREDIVIDE_FACTOR_FORMAT = '''
Gradient predivide factor should be enabled as:
"gradient_predivide_factor": 1.0
'''
GRADIENT_PREDIVIDE_FACTOR = "gradient_predivide_factor"
GRADIENT_PREDIVIDE_FACTOR_DEFAULT = 1.0


#########################################
# Dump DeepSpeed state
#########################################
# Dump State. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DUMP_STATE_FORMAT = '''
Dump state should be enabled as:
"dump_state": true
'''
DUMP_STATE = 'dump_state'
DUMP_STATE_DEFAULT = False

#########################################
# Vocabulary size
#########################################
# Vocabulary size.
# Users can configure in ds_config.json as below example:
VOCABULARY_SIZE_FORMAT = '''
Vocabulary size can be specified as:
"vocabulary_size": 1024
'''
VOCABULARY_SIZE = 'vocabulary_size'
VOCABULARY_SIZE_DEFAULT = None

#########################################
# Wall block breakdown
#########################################
# Wall clock breakdown. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
WALL_CLOCK_BREAKDOWN_FORMAT = '''
Wall block breakdown should be enabled as:
"wall_clock_breakdown": true
'''
WALL_CLOCK_BREAKDOWN = 'wall_clock_breakdown'
WALL_CLOCK_BREAKDOWN_DEFAULT = False


#########################################
# Tensorboard
#########################################
# Tensorboard. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
TENSORBOARD_FORMAT = '''
Tensorboard can be specified as:
"tensorboard": {
  "enabled": true,
  "output_path": "/home/myname/foo",
  "job_name": "model_lr2e-5_epoch3_seed2_seq64"
}
'''
TENSORBOARD = "tensorboard"

# Tensorboard enable signal
TENSORBOARD_ENABLED = "enabled"
TENSORBOARD_ENABLED_DEFAULT = False

# Tensorboard output path
TENSORBOARD_OUTPUT_PATH = "output_path"
TENSORBOARD_OUTPUT_PATH_DEFAULT = ""

# Tensorboard job name
TENSORBOARD_JOB_NAME = "job_name"
TENSORBOARD_JOB_NAME_DEFAULT = "MerakJobName"

class DeepSpeedConfigError(Exception):
    pass


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict,
                            GRADIENT_ACCUMULATION_STEPS,
                            GRADIENT_ACCUMULATION_STEPS_DEFAULT)

def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)

def get_gradient_predivide_factor(param_dict):
    return get_scalar_param(param_dict,
                            GRADIENT_PREDIVIDE_FACTOR,
                            GRADIENT_PREDIVIDE_FACTOR_DEFAULT)

def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)

def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


def get_gradient_clipping(param_dict):
    return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


def get_pipeline_config(param_dict):
    '''Parses pipeline engine configuration. '''
    default_pipeline = {
        'stages': 'auto',
        'partition': 'best',
        'seed_layers': False,
        'activation_checkpoint_interval': 0
    }
    config = default_pipeline
    for key, val in param_dict.get('pipeline', {}).items():
        config[key] = val
    return config


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(param_dict,
                            TRAIN_MICRO_BATCH_SIZE_PER_GPU,
                            TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict,
                            WALL_CLOCK_BREAKDOWN,
                            WALL_CLOCK_BREAKDOWN_DEFAULT)

def get_tensorboard_enabled(param_dict):
    if TENSORBOARD in param_dict.keys():
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_ENABLED,
                                TENSORBOARD_ENABLED_DEFAULT)
    else:
        return False


def get_tensorboard_output_path(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_OUTPUT_PATH,
                                TENSORBOARD_OUTPUT_PATH_DEFAULT)
    else:
        return TENSORBOARD_OUTPUT_PATH_DEFAULT


def get_tensorboard_job_name(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_JOB_NAME,
                                TENSORBOARD_JOB_NAME_DEFAULT)
    else:
        return TENSORBOARD_JOB_NAME_DEFAULT


'''Write deepspeed config files by modifying basic templates.
Can be used for quicly changing parameters via command line parameters.'''


class DeepSpeedConfigWriter:
    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def add_config(self, key, value):
        self.data[key] = value

    def load_config(self, filename):
        self.data = json.load(open(filename,
                                   'r'),
                              object_pairs_hook=dict_raise_error_on_duplicate_keys)

    def write_config(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.data, outfile)


class DeepSpeedConfig(object):
    def __init__(self, config: Union[str, dict], mpu=None):
        super(DeepSpeedConfig, self).__init__()
        if isinstance(config, dict):
            self._param_dict = config
        elif os.path.exists(config):
            self._param_dict = json.load(
                open(config,
                     'r'),
                object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            raise ValueError(
                f"Expected a string path to an existing deepspeed config, or a dictionary. Received: {config}"
            )
        try:
            self.global_rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            if mpu is None:
                self.dp_world_size = torch.distributed.get_world_size()
            else:
                self.dp_world_size = mpu.get_data_parallel_world_size()
        except:
            self.global_rank = 0
            self.dp_world_size = self.world_size = 1

        self._initialize_params(self._param_dict)
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(
            param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.gradient_predivide_factor = get_gradient_predivide_factor(param_dict)
        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.wall_clock_breakdown = get_wall_clock_breakdown(
              param_dict)
        self.tensorboard_enabled = get_tensorboard_enabled(param_dict)
        self.tensorboard_output_path = get_tensorboard_output_path(param_dict)
        self.tensorboard_job_name = get_tensorboard_job_name(param_dict)

    def _batch_assertion(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        assert train_batch > 0, \
            f'Train batch size: {train_batch} has to be greater than 0'

        assert micro_batch > 0, \
            f'Micro batch size per gpu: {micro_batch} has to be greater than 0'

        assert grad_acc > 0, \
            f'Gradient accumulation steps: {grad_acc} has to be greater than 0'

        assert train_batch == micro_batch * grad_acc * self.dp_world_size, \
                (f'Check batch related parameters. train_batch_size is not equal'
                ' to micro_batch_per_gpu * gradient_acc_step * world_size'
                f'{train_batch} != {micro_batch} * {grad_acc} * {self.dp_world_size}')

    def _set_batch_related_parameters(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        #all values are provided nothing needs to be set
        if train_batch is not None and \
            micro_batch is not None and \
            grad_acc is not None:
            return

        #global_accumulation_steps needs to be set
        elif train_batch is not None and \
            micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.dp_world_size
            self.gradient_accumulation_steps = grad_acc

        #micro_batch_per_gpu needs to be set
        elif train_batch is not None and \
            grad_acc is not None:
            micro_batch = train_batch // self.dp_world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch

        #train_batch_size needs to be set
        elif micro_batch is not None and \
            grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.dp_world_size
            self.train_batch_size = train_batch_size

        #gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.dp_world_size

        #train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.dp_world_size
            self.gradient_accumulation_steps = 1

        #either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, \
                'Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided'

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()
        self._do_warning_check()

    def print(self, name):
        logger.info('{}:'.format(name))
        for arg in sorted(vars(self)):
            if arg != '_param_dict':
                dots = '.' * (29 - len(arg))
                logger.info('  {} {} {}'.format(arg, dots, getattr(self, arg)))


    def _do_error_check(self):
        assert self.train_micro_batch_size_per_gpu, "DeepSpeedConfig: {} is not defined".format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)

        assert self.gradient_accumulation_steps, "DeepSpeedConfig: {} is not defined".format(
            GRADIENT_ACCUMULATION_STEPS)

    def _do_warning_check(self):

        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logger.warning(
                "DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization."
                .format(vocabulary_size,
                        TENSOR_CORE_ALIGN_SIZE))
