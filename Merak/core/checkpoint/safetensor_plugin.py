import os
import json
import datetime
import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from torch.distributed import get_rank, get_world_size, barrier

from Merak import print_rank_0

from .checkpoint import ensure_directory_exists
from .. import mpu
from ..recompute import get_rng_tracker

attr_dict = {
    "row": 1,
    "col": 0,
}

def get_merak_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from Merak checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        if tp_size == 1 and pp_size == 1:
            sub_dir_name = ""
        else:
            sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_pp_rank_{pp_rank:03d}"
        for checkpoint_name in ["partial_model_optim.pt", "model_optim.pt"]:
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            if os.path.isfile(checkpoint_path):
                break
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts

def get_parallel_degree(dirlist):
    if "pp_rank" in dirlist[0]:
        degree_re = re.compile(r"mp_rank\_([0-9_.]+)\_([a-z]+)\_([a-z]+)\_([0-9_.]+)")
    else:
        degree_re = re.compile(r"mp_rank\_([a-z0-9_]+)")

    pp_list = []
    tp_list = []
    for sub_dir in dirlist:
        degree = degree_re.match(sub_dir)
        tp = degree.group(1)
        tp_list.append(tp)
        if "pp_rank" in sub_dir:
            pp = degree.group(4)
            pp_list.append(pp)
    tp_size = int(max(tp_list)) + 1
    if pp_list:
        pp_size = int(max(pp_list)) + 1
    else:
        pp_size = 1
    return tp_size, pp_size

def save_3d_parallel_model(model, args, iteration):
    """
    分布式保存3D并行模型
    每个进程仅保存自己负责的参数分片
    Args:
        model: 3D并行的模型 (如Megatron-LM结构)
        save_path: 保存路径（自动添加进程编号后缀）
    """

    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_pp_rank_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    if "model_optim.pt" in sub_dirs:
        rank0_checkpoint_path = os.path.join(args.load_path, "model_optim.pt")
    print(f"Loading Merak checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")

    # Get parallel degree
    if "model_optim.pt" in sub_dirs:
        tp_size,  pp_size = (1, 1)
    else:
        tp_size,  pp_size = get_parallel_degree(sub_dirs)


    dtime = datetime.datetime.now().strftime('%Y-%m-%d')
    save_path = args.output_dir+'/{time}_ckpt'.format(time=dtime)
    index_file_path = os.path.join(save_path, "model.safetensors.index.json")

    pp_rank = mpu.get_pipe_parallel_rank()
    pp_size = mpu.get_pipe_parallel_world_size()
    tp_rank = mpu.get_model_parallel_rank()
    tp_size = mpu.get_model_parallel_world_size()

    if mpu.get_data_parallel_rank() == 0:
        state_dict = {}
        model_state_dict = {}
        weight_map = {}
        tp_meta_data = {}

        if dist.get_rank() == 0:
            ensure_directory_exists(index_file_path)

        for n, p in model.state_dict().items():
            n = ".".join(n.split(".")[1:])
            end_point = n.split(".")[-1]
            if "col" in n:
                gather_list = [
                    torch.empty_like(p).to(p.device)
                    for _ in range(tp_size)
                ]
                gather_list[tp_rank] = p
                dist.all_gather(gather_list, p, group=mpu.get_model_parallel_group())
                gather_list = [tensor.clone().detach().cpu() for tensor in gather_list]
                dim = 0 if "bias" == end_point else attr_dict["col"]
                full_param = torch.cat(gather_list, dim=dim)
                param_name = n.replace("col_linear.", "")
                tp_meta_data[n] = dim
                model_state_dict[param_name] = full_param
            elif "row" in n:
                if "bias" == end_point:
                    param_name = n.replace("row_linear.", "")
                    model_state_dict[param_name] = p
                    continue
                gather_list = [
                    torch.empty_like(p).to(p.device)
                    for _ in range(tp_size)
                ]
                gather_list[tp_rank] = p
                dist.all_gather(gather_list, p, group=mpu.get_model_parallel_group())
                gather_list = [tensor.clone().detach().cpu() for tensor in gather_list]
                dim = attr_dict["row"]
                full_param = torch.cat(gather_list, dim=dim)
                param_name = n.replace("row_linear.", "")
                tp_meta_data[n] = dim
                model_state_dict[param_name] = full_param
            elif ("wte" in n or "tied_modules" in n) and args.parallel_vocab:
                gather_list = [
                    torch.empty_like(p).to(p.device)
                    for _ in range(tp_size)
                ]
                gather_list[tp_rank] = p
                dist.all_gather(gather_list, p, group=mpu.get_model_parallel_group())
                gather_list = [tensor.clone().detach().cpu() for tensor in gather_list]
                full_param = torch.cat(gather_list, dim=0)
                tp_meta_data[n] = 0
                model_state_dict[n] = full_param
            else:
                gather_list = None
                model_state_dict[n] = p

        if pp_size == 1:
            save_name = "model.safetensors"
        else:
            save_name = f"model-{pp_rank:05d}-of-{pp_size:06d}.safetensors"
        
        for k in model_state_dict.keys():
            weight_map[k] = save_name

        index_dict = {
            "metadata": {
                "total_size": {save_name: sum(t.numel() * torch.finfo(t.dtype).bits // 8 
                                for t in model.parameters())},
                "format": "pt",
                "iteration": iteration,
            },
            "tp_meta_data": tp_meta_data,
            "weight_map": weight_map
        }

        if tp_rank == 0:
            for rank in range(pp_size):
                if rank == pp_rank:
                    save_file(
                        model_state_dict,
                        os.path.join(save_path, save_name)
                    )
                else:
                    continue
                if pp_rank == 0:
                    with open(index_file_path, "w") as f:
                        json.dump(index_dict, f, indent=2)
                else:
                    with open(index_file_path, "r") as f:
                        before_index_dict = json.load(f)
                    before_index_dict["metadata"]["total_size"].update(
                        index_dict["metadata"]["total_size"]
                    )
                    before_index_dict["tp_meta_data"].update(index_dict["tp_meta_data"])
                    before_index_dict["weight_map"].update(index_dict["weight_map"])
                    with open(index_file_path, "w") as f:
                        json.dump(before_index_dict, f, indent=2)
        dist.barrier()

        del gather_list
        del model_state_dict



        # state_dict['args'] = args
        # state_dict['iteration'] = iteration
        # state_dict['model'] = model_state_dict
