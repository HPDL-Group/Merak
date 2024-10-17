# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/schedule.py

from .exec_schedule import *

class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.is_first_stage or self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))

            if self._valid_micro_batch(micro_batch_id):
                cmds.append(ForwardPass(recv_buf))

            yield cmds

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        return 2


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also
            # whether it is a forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                    self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and \
                    self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and \
                    self._valid_stage(self.next_stage):
                    cmds.append(SendActivation(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and \
                    self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """As many buffers as the distance from this stage to the last stage.
        """
        buffers = min(self.stages - self.stage_id + 1, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class MergeP2PTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also
            # whether it is a forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                    self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and \
                        self._valid_stage(self.prev_stage):
                        cmds.append(SendGradRecvActivation((prev_buffer,
                                                            curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and \
                    self._valid_stage(self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and \
                    self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and \
                        self._valid_stage(self.next_stage):
                        cmds.append(SendActivationRecvGrad((prev_buffer,
                                                            curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and \
                    self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

class PreRecomputeTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also
            # whether it is a forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                   self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and \
                       self._valid_stage(self.prev_stage):
                        cmds.append(SendGradRecvActivation((prev_buffer,
                                                            curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and \
                     self._valid_stage(self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and \
                   self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and \
                       self._valid_stage(self.next_stage):
                        cmds.append(SendActivationRecvGrad((prev_buffer,
                                                            curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and \
                     self._valid_stage(self.next_stage):
                    cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id \
                       < self.stages:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class LastNoRecomputeTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        frist_sendrecv = True
        frist_fp = True
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also
            # whether it is a forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                   self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id) and \
                       self._valid_stage(self.prev_stage):
                        if self.stage_id == self.stages - 1 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        else:
                            cmds.append(SendGradRecvActivation((prev_buffer,
                                                                curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and \
                     self._valid_stage(self.prev_stage):
                        cmds.append(SendGrad(prev_buffer))
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and \
                   self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and \
                       self._valid_stage(self.next_stage):
                        if self.stage_id == self.stages - 2 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(SendActivation(prev_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendActivationRecvGrad((prev_buffer,
                                                                curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and \
                     self._valid_stage(self.next_stage):
                    if self.stage_id == self.stages - 2:
                        cmds.append(RecvGrad(curr_buffer))
                    else:
                        cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id \
                        < self.stages and \
                        self.stage_id != self.stages - 2:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    elif self.stage_id == self.stages - 2 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

class FullCriticalPathTrainSchedule(TrainSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """
    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        frist_sendrecv = True
        second_sendrecv = True
        frist_fp = True
        second_fp = True
        frist_bp = 0
        b0_buffer = self._buffer_idx(0)
        f2_buffer = self._buffer_idx(2)

        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also
            # whether it is a forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(micro_batch_id) and \
                   self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(prev_micro_batch_id):
                        if self.stage_id == self.stages - 1 and frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        elif self.stage_id == self.stages - 1 and \
                             second_sendrecv:
                            second_sendrecv = False
                            cmds.append(SendGrad(prev_buffer))
                            cmds.append(RecvActivation(curr_buffer))
                        elif self.stage_id == self.stages - 2 and \
                             frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(RecomputeRecvGrad(b0_buffer))
                            cmds.append(SendActivation(f2_buffer-1))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendGradRecvActivation((prev_buffer,
                                                                curr_buffer)))
                    else:
                        cmds.append(RecvActivation(curr_buffer))
                elif self._valid_micro_batch(prev_micro_batch_id) and \
                     self._valid_stage(
                            self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                elif frist_sendrecv and self.stages == 2 and \
                     self._valid_micro_batch(micro_batch_id) and \
                     self.stage_id == 0 and \
                     self._valid_micro_batch(prev_micro_batch_id):
                    frist_sendrecv = False
                    cmds.append(RecomputeRecvGrad(b0_buffer))
                    cmds.append(SendActivation(f2_buffer-1))
                    cmds.append(RestoreRecomputeStatus())
            else:
                if self._valid_micro_batch(prev_micro_batch_id) and \
                   self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id) and \
                       self._valid_stage(self.next_stage):
                        if self.stage_id == self.stages - 2 and frist_sendrecv:
                            cmds.append(RecvActivation(f2_buffer))
                        elif self.stage_id == self.stages - 2 and \
                             second_sendrecv:
                            second_sendrecv = False
                            cmds.append(SendGrad(b0_buffer))
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(SendActivation(f2_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        elif self.stage_id == self.stages - 3 and \
                             frist_sendrecv:
                            frist_sendrecv = False
                            cmds.append(SendActivation(prev_buffer))
                            cmds.append(RecomputeRecvGrad(curr_buffer))
                            cmds.append(RestoreRecomputeStatus())
                        else:
                            cmds.append(SendActivationRecvGrad((prev_buffer,
                                                                curr_buffer)))
                    else:
                        cmds.append(SendActivation(prev_buffer))
                elif self._valid_micro_batch(micro_batch_id) and \
                     self._valid_stage(
                        self.next_stage):
                    if self.stage_id == self.stages - 2:
                        cmds.append(RecvGrad(curr_buffer))
                    else:
                        cmds.append(RecomputeRecvGrad(curr_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    if self.micro_batches - micro_batch_id + self.stage_id \
                        < self.stages and \
                        self.stage_id != self.stages - 2:
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                    elif self.stage_id == self.stages - 2 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    elif self.stage_id == self.stages - 2 and second_fp:
                        second_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    elif self.stage_id == self.stages - 2 and frist_bp == 1:
                        # use third forward for first backward 

                        frist_bp = 2
                        if isinstance(cmds[-1], LoadMicroBatch):
                            cmds[-1] = BackwardPass(b0_buffer)
                        else:
                            cmds.append(BackwardPass(b0_buffer))

                    elif self.stage_id == self.stages - 3 and frist_fp:
                        frist_fp = False
                        cmds.append(PreCheckpointForwardPass(curr_buffer))
                        cmds.append(RestoreRecomputeStatus())
                    else:
                        cmds.append(ForwardPass(curr_buffer))
                else:
                    if self.stage_id == self.stages - 2 and frist_bp == 0:
                        # use first backward for third forward 
                        frist_bp = 1
                        if self.stage_id == 0:
                            cmds.append(LoadMicroBatch(f2_buffer))
                        cmds.append(ForwardPass(f2_buffer))
                    else:
                        cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                if self.stage_id < self.stages - 1:
                    cmds.append(RestoreRecomputeStatus())
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with
    gradient accumulation.
    """
    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                LoadMicroBatch(buffer_id=0),
                ForwardPass(buffer_id=0),
                BackwardPass(buffer_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend([
                    ReduceGrads(),
                    OptimizerStep(),
                ])
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed.
        """
        return 1


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
