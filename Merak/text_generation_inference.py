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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/v2.3/megatron/text_generation_utils.py

from . import MerakTrainer, mpu

import torch
import torch.nn.functional as F
import torch.distributed as dist
import os

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits

def switch(val1, val2, boolean):

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2

class text_generation_pipeline(MerakTrainer):
    def __init__(self, **kwargs):
        self.input = None
        super().__init__(**kwargs)

    def get_batch(self, context_tokens):
        if self.input is not None:
            context_tokens = self.input
        tokens = context_tokens.view(self.args.per_device_train_batch_size, -1).contiguous().cuda()
        # Get the attention mask.
        attention_mask = torch.ones((self.args.seq_length), device=tokens.device)
        return (tokens, attention_mask)

    def sample_sequence_batch(self, context_tokens, context_lengths,
                        maxlen=None, type_ids=None):

        args = self.args
        tokenizer = self.tokenizer

        args.top_p = 0.9
        args.top_k = 0
        tp_size = mpu.get_model_parallel_world_size()
        pp_size = mpu.get_pipe_parallel_world_size()

        self.pipe_model.eval()
        with torch.no_grad():
            context_length = context_lengths.min().item()

            # added eos_id to support the function generate_samples_eval that passes
            # eos_id as an argument and needs termination when that id id found.
            if hasattr(args, 'eos_id'):
                eos_id = args.eos_id
            else:
                eos_id = tokenizer.eod

            counter = 0
            org_context_length = context_length

            layer_past = None
            batch_size = context_tokens.size(0)
            is_done = torch.zeros([batch_size]).byte().cuda()
            tokens = context_tokens
            self.input = tokens
            if maxlen is None:
                maxlen = args.seq_length - 1
                if maxlen > (org_context_length + args.out_seq_length):
                    maxlen = org_context_length + args.out_seq_length

            lengths = torch.ones([batch_size]).long().cuda() * maxlen
            while context_length <= (maxlen):
                loss, output, labels = self.pipe_model.eval_batch(batch_fn=self.get_batch)

                if output is not None and tp_size > 1:
                    tensor_list = [torch.zeros([1, args.seq_length,
                                                self.pipe_model.config.vocab_size // tp_size]).cuda()
                                                for _ in range(tp_size)]
                    dist.all_gather(tensor_list, output[0][0], group=mpu.get_model_parallel_group())
                    output = (torch.cat(tuple(tensor_list), dim=-1),)

                if dist.get_rank() == dist.get_world_size() - 1:
                    assert output is not None
                    logits = output[0][:, context_length - 1, :]

                # if mpu.is_pipeline_last_stage():
                    if False: #args.greedy:
                        prev = torch.argmax(logits, dim=-1).view(-1)
                    else:
                        logits = logits.float()
                        logits /= args.temperature
                        # logits = top_k_logits(logits, top_k=args.top_k,
                        #                       top_p=args.top_p)
                        log_probs = F.softmax(logits, dim=-1)
                        prev = torch.multinomial(log_probs, num_samples=1).view(-1)  # 获取输入tensor中最大值的index，因为样本数为1
                    started = context_lengths <= context_length

                    new_tokens = switch(
                        tokens[:, context_length].view(-1), prev, started)
                    tokens[:, context_length] = new_tokens
                    src = dist.get_world_size() - 1
                    # group = mpu.get_pipe_parallel_group()
                    torch.distributed.broadcast(new_tokens, src)

                    done_token = (prev == eos_id).byte() & started.byte()
                    just_finished = (done_token & ~is_done).bool()
                    lengths[just_finished.view(-1)] = context_length
                    is_done = is_done | done_token

                    done = torch.all(is_done)
                    src = dist.get_world_size() - 1
                    # group = mpu.get_pipe_parallel_group()
                    torch.distributed.broadcast(done, src)
                    self.input = tokens
                    yield tokens, lengths

                else:
                    # if mpu.is_pipeline_first_stage():
                    src = dist.get_world_size() - 1
                    # group = mpu.get_pipe_parallel_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src)
                    tokens[:, context_length] = new_tokens
                    self.input = tokens
                    yield tokens, None

                    done = torch.cuda.ByteTensor([0])
                    src = dist.get_world_size() - 1
                    # group = mpu.get_pipe_parallel_group()
                    torch.distributed.broadcast(done, src)
                context_length += 1
                counter += 1
                if done:
                    break

    def generate_samples_interactive(self, print_frequency=24):

        self.create_optimizer_and_scheduler(num_training_steps=1)

        # load merak checkpoint
        if self.args.resume_from_checkpoint and os.path.isdir(self.args.resume_from_checkpoint):
            iteration = self.load_from_checkpoint(self.args.resume_from_checkpoint)
        else:
            raise ValueError(f"{self.args.resume_from_checkpoint} has no checkpoints, please check the path or dirctory")


        args = self.args
        tokenizer = self.tokenizer

        context_count = 0
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
            and mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    # context_tokens = tokenizer.tokenize(raw_text)
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_tokens = tokenizer.convert_tokens_to_ids(context_tokens)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                            "\nPlease give smaller context (half of the "
                            "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_tokens = tokenizer.convert_tokens_to_ids(context_tokens)
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(input_info_tensor)
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            # if mpu.get_model_parallel_rank() == 0 \
            if mpu.get_pipe_parallel_world_size() > 1:
                if mpu.is_pipeline_first_stage() and mpu.get_model_parallel_rank() == 0:
                    src = 0 #self.pipe_model.grid.stage_to_global(stage_id=0)
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src)
                else:
                    src = 0 #self.pipe_model.grid.stage_to_global(stage_id=0)
                    context_tokens_tensor = torch.empty(context_length,
                                                        dtype=torch.int64,
                                                        device=torch.device("cuda"))
                    torch.distributed.broadcast(context_tokens_tensor, src)
                    torch.cuda.synchronize()
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = self.get_token_stream([context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if counter % print_frequency != 0 \
                or mpu.get_model_parallel_rank() != 0 \
                or not mpu.is_pipeline_first_stage():
                    continue

                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

                # trim_decode_tokens = tokenizer.detokenize(
                #     decode_tokens)[raw_text_len:]
                trim_decode_tokens = tokenizer.decode(
                    decode_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if mpu.is_pipeline_first_stage() \
            and mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nContext:", raw_text, flush=True)

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.decode(
                    decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1

    def get_token_stream(self, context_tokens):
        args = self.args
        tokenizer = self.tokenizer

        context_tokens, context_lengths = self.pad_batch(context_tokens,
                                                    tokenizer.eod, args)

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        context_length_tensor = torch.cuda.LongTensor(context_lengths)

        torch.distributed.broadcast(context_length_tensor,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())

        context_length = context_length_tensor.min().item()

        torch.distributed.broadcast(context_tokens_tensor, 0)

        tokens, attention_mask = self.get_batch(context_tokens_tensor)

        batch_token_iterator = self.sample_sequence_batch(context_tokens_tensor,
                                                    context_length_tensor)
        for tokens, lengths in batch_token_iterator:
            context_length += 1
            if tokens is not None:
                yield tokens[:, :context_length], lengths
            else:
                yield None, None

    def pad_batch(self, batch, pad_id, args):

        context_lengths = []
        for tokens in batch:
            context_length = len(tokens)
            if context_length < args.seq_length:
                tokens.extend([pad_id] * (args.seq_length - context_length))
            context_lengths.append(context_length)
        return batch, context_lengths