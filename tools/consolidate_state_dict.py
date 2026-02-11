import argparse
import os
import torch

from collections import defaultdict
from natsort import natsorted

# python consolidate_state_dict.py /path/to/iteration_checkpoints /
# add --zero_reshard /path/to/consolidate_checkpoint.pt to enable zero reshard


def merge_zero_stage1_optimizer_states_from_paths(dir_path: str, decay: float):
    pt_paths = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".pt") and f.startswith("zero")
    ]

    pt_paths = natsorted(pt_paths)
    for path in pt_paths:
        print(path)

    checkpoint_list = []
    for path in pt_paths:
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_list.append(checkpoint)

    return merge_zero_stage1_optimizer_states(checkpoint_list, decay)


def merge_zero_stage1_optimizer_states(checkpoint_list, decay):
    """
    Merge optimzier state dict for zero-1
    """
    merged_state = {}
    decay_group = {"params": [], "weight_decay": decay}
    no_decay_group = {"params": [], "weight_decay": 0.0}
    global_param_id = 0

    loss_scaler = None
    dynamic_loss_scale = None
    overflow = None


    state_accumulator = defaultdict(lambda: defaultdict(list))
    param_group_accumulator = defaultdict(list)  # key: local_param_id → ['decay' / 'nodecay']

    for idx, ckpt in enumerate(checkpoint_list):
        base_opt_state = ckpt["optimizer_state_dict"]["base_optimizer_state"]

        if idx == 0:
            loss_scaler = ckpt["optimizer_state_dict"].get("loss_scaler")
            dynamic_loss_scale = ckpt["optimizer_state_dict"].get("dynamic_loss_scale")
            overflow = ckpt["optimizer_state_dict"].get("overflow")

        state = base_opt_state["state"]
        param_groups = base_opt_state["param_groups"]

        for group in param_groups:
            weight_decay = group.get("weight_decay", 0.0)
            group_type = "decay" if weight_decay > 0 else "nodecay"

            for local_param_id in group["params"]:
                param_state = state[local_param_id]
                for field_name, value in param_state.items():
                    state_accumulator[local_param_id][field_name].append(value)
                param_group_accumulator[local_param_id].append(group_type)

    # concat state and reconstruct global_param_id
    for local_param_id in sorted(state_accumulator.keys()):
        state_fields = state_accumulator[local_param_id]
        merged_param_state = {}

        for field_name, value_list in state_fields.items():
            if isinstance(value_list[0], torch.Tensor):
                if value_list[0].ndim > 0:
                    merged_param_state[field_name] = torch.cat(value_list, dim=0)
                else:
                    # ensure all are tensors before stacking
                    tensor_list = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in value_list]
                    merged_param_state[field_name] = torch.stack(tensor_list).max()
            else:
                merged_param_state[field_name] = value_list[0]


        merged_state[global_param_id] = merged_param_state

        # distribute param into decay and no_decay groups
        group_type = param_group_accumulator[local_param_id][0]
        if group_type == "decay":
            decay_group["params"].append(global_param_id)
        else:
            no_decay_group["params"].append(global_param_id)

        global_param_id += 1

    return {
        "state": merged_state,
        "param_groups": [decay_group, no_decay_group],
        "loss_scaler": loss_scaler,
        "dynamic_loss_scale": dynamic_loss_scale,
        "overflow": overflow,
    }


# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="List of .pt checkpoint files to merge (e.g., checkpoint_rank0.pt checkpoint_rank1.pt ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for merged optimizer state",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.00,
        help="weight_decay value used when initializing the optimizer.",
    )
    return parser


def main():
    # args
    parser = argparse.ArgumentParser(
        description="Merge ZeRO Stage 1 optimizer states from .pt files"
    )
    parser = parse_option(parser)
    args = parser.parse_args()
    pt_paths = args.checkpoints
    decay = args.decay

    # merge state dict
    merged_state_dict = merge_zero_stage1_optimizer_states_from_paths(pt_paths, decay)

    # save merged optimizer state dict
    if args.output is None:
        output_path = pt_paths + "/consolidated_opt_state_dict.pt"
    else:
        output_path = args.output
    torch.save(merged_state_dict, output_path)

    print(f"Saved merged optimizer state to: {output_path}")


if __name__ == "__main__":
    main()
