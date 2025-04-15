import os

import torch.optim as optim

from functools import partial
from argparse import ArgumentParser

from pytorch_unet.unet.unet import UNet2D
from pytorch_unet.unet.model import Model
from pytorch_unet.unet.utils import MetricList
from pytorch_unet.unet.metrics import jaccard_index, f1_score, LogNLLLoss
from pytorch_unet.unet.dataset import JointTransform2D, ImageToImage2D, Image2D

import torch
import Merak
from Merak import MerakArguments, MerakTrainer
from transformers import HfArgumentParser

def parse_option(parser):
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--out_channels', default=3, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--width', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_freq', default=0, type=int)
    parser.add_argument('--save_model', default=0, type=int)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--crop', type=int, default=None)
    return parser

pp = 2
tp = 1
dp = 1
Merak.init(pp, tp, dp)

# merge args
hfparser = HfArgumentParser(MerakArguments)
parser = parse_option(hfparser)
training_args, args = parser.parse_args_into_dataclasses()

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.train_dataset, tf_val)
predict_dataset = Image2D(args.train_dataset)

conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
unet = UNet2D(args.in_channels, args.out_channels, conv_depths)
training_args.num_layers = args.depth
# loss = LogNLLLoss()
# optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

results_folder = os.path.join(args.checkpoint_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

class UnetTrainer(MerakTrainer):

    def create_optimizer(self, model):
        self.optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)

        return self.optimizer
    
    def prepare_data(
        self,
        data
        ):
        if not isinstance(data, (tuple, list)):
            if hasattr(data, "data"):
                data = data.data
            if isinstance(data, dict):
                inputs_list = []
                for key, val in self.input_to_stage_dic.items():
                    for d in list(data.keys()):
                        for i in val:
                            if d in i:
                                inputs_list.append(data.pop(d))
                                break
                inputs_list += list(data.values())
                return tuple(inputs_list)
            else:
                raise NotImplementedError('only support data in tuple, list or dict')
        else:
            data = data[:-1]
            return data

    def get_loss_fn(self):
        criterion = LogNLLLoss()
        def loss_fn(
            outputs,
            labels
            ):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            labels = torch.argmax(labels, dim=3)
            # print(outputs.size(), labels.size())
            loss = criterion(outputs,
                             labels)
            return loss
        return loss_fn

trainer = UnetTrainer(
    model=unet,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
