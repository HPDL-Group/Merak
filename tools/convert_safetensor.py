import argparse
import datetime
import json
import logging
import os
import re
from typing import Tuple

import torch
from safetensors.torch import load_file, save_file

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

attr_dict = {
    "row": 1,
    "col": 0,
}


def ensure_directory_exists(filename: str) -> None:
    """确保文件所在的目录存在，如果不存在则创建目录."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        logger.info(f"Create directory: {dirname}")


def get_merak_sharded_states(
    args: argparse.Namespace, tp_size: int, pp_size: int, pp_rank: int
) -> list:
    """
    从Merak检查点加载分片状态字典, 基于tensor并行大小、pipeline并行大小和rank。

    Args:
        args (argparse.Namespace): 脚本的命令行参数
        tp_size (int): tensor并行大小
        pp_size (int): pipeline并行大小
        pp_rank (int): pipeline并行rank

    Returns:
        list: 包含tp_size个Tensor并行的状态字典
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = (
            f"mp_rank_{i:02d}"
            if pp_size == 1
            else f"mp_rank_{i:02d}_pp_rank_{pp_rank:03d}"
        )
        checkpoint_name = (
            "partial_model_optim.pt"
            if os.path.isfile(
                os.path.join(args.load_path, sub_dir_name, "partial_model_optim.pt")
            )
            else "model_optim.pt"
        )
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file does not exist: {checkpoint_path}"
            )
        try:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            tp_state_dicts.append(state_dict)
        except Exception as e:
            logger.error(f"Failed to load the checkpoint: {checkpoint_path}")
            raise e
    return tp_state_dicts


def get_parallel_degree(dirlist: list) -> Tuple[int, int]:
    """
    从目录列表中推断并行程度.

    Args:
        dirlist (list): 包含分片检查点的目录列表

    Returns:
        tuple[int, int]: tensor并行大小和pipeline并行大小
    """
    dirlist = [s for s in dirlist if "mp_rank" in s]
    if not dirlist:
        raise ValueError(
            "Unable to determine the degree of parallelism: Not valid directories found"
        )

    if "pp_rank" in dirlist[0]:
        degree_re = re.compile(r"mp_rank_([0-9_]+)_pp_rank_([0-9_]+)")
    else:
        degree_re = re.compile(r"mp_rank_([a-z0-9_]+)")

    tp_list = []
    pp_list = []
    for sub_dir in dirlist:
        degree_match = degree_re.match(sub_dir)
        if not degree_match:
            continue
        tp_rank = degree_match.group(1)
        tp_list.append(int(tp_rank))
        if "pp_rank" in sub_dir:
            pp_rank = degree_match.group(2)
            pp_list.append(int(pp_rank))

    tp_size = max(tp_list) + 1 if tp_list else 1
    pp_size = max(pp_list) + 1 if pp_list else 1
    logger.info(f"Find tp_size={tp_size}, pp_size={pp_size}")

    return tp_size, pp_size


def convert_safetensor_ckpt(args: argparse.Namespace) -> None:
    """
    将Merak格式的检查点文件转换为safetensors格式.
    """
    load_path = args.load_path
    save_path = args.save_path

    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Checkpoint files not exit: {load_path}")

    sub_dirs = os.listdir(load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_pp_rank_000"]

    # 寻找有效的子目录以确定检查点路径
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_dir = sub_dir
            break
    else:
        if "model_optim.pt" in sub_dirs:
            rank0_dir = ""
        else:
            raise FileNotFoundError("No valid checkpoint file found")

    if rank0_dir:
        checkpoint_name = os.listdir(os.path.join(load_path, rank0_dir))[0]
        rank0_checkpoint_path = os.path.join(load_path, rank0_dir, checkpoint_name)
    else:
        rank0_checkpoint_path = os.path.join(load_path, "model_optim.pt")

    logger.info(f"Loading Merak checkpoint from path: {rank0_checkpoint_path}")
    try:
        state_dict = torch.load(
            rank0_checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as e:
        logger.error(f"Failed to load the checkpoint: {rank0_checkpoint_path}")
        raise e

    # 获取或使用默认的parallel_vocab参数
    merak_args = state_dict.get("args", None)
    parallel_vocab = (
        merak_args.parallel_vocab if merak_args is not None else args.parallel_vocab
    )

    # 计算并行度
    if "model_optim.pt" in sub_dirs:
        tp_size, pp_size = (1, 1)
    else:
        tp_size, pp_size = get_parallel_degree(sub_dirs)

    # 时间戳用于保存文件
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.save_path, f"{current_time}_ckpt")
    os.makedirs(save_path, exist_ok=True)

    index_file_path = os.path.join(save_path, "model.safetensors.index.json")
    ensure_directory_exists(index_file_path)

    for pp_rank in range(pp_size):
        logger.info(f"Deal with pipeline parallel rank: {pp_rank}")
        weight_map = {}
        tp_meta_data = {}
        model_state_dict = {}
        total_size = 0

        if pp_size > 0:
            tp_state_dicts = get_merak_sharded_states(args, tp_size, pp_size, pp_rank)

        for tp_rank in range(tp_size):
            logger.debug(f"Deal with tensor parallel rank: {tp_rank}")
            state_dict = tp_state_dicts[tp_rank]
            for n, p in state_dict["model"].items():
                if "col" in n:
                    # 处理列并行的参数
                    gather_list = [
                        tp_state_dicts[r]["model"][n] for r in range(tp_size)
                    ]
                    if len(gather_list) != tp_size:
                        logger.warning(
                            f"参数 {n} 的分片数量 {len(gather_list)} 不等于 tp_size {tp_size}"
                        )
                        continue
                    dim = 0
                    full_param = torch.cat(gather_list, dim=dim)
                    tp_meta_data[n] = dim
                    param_name = ".".join(n.split(".")[1:]).replace("col_linear.", "")
                    model_state_dict[param_name] = full_param
                    total_size += full_param.numel()
                elif "row" in n:
                    # 处理行并行的参数
                    if "bias" in n.split("."):
                        param_name = ".".join(n.split(".")[1:]).replace(
                            "row_linear.", ""
                        )
                        model_state_dict[param_name] = p
                        continue
                    gather_list = [
                        tp_state_dicts[r]["model"][n] for r in range(tp_size)
                    ]
                    if len(gather_list) != tp_size:
                        logger.warning(
                            f"参数 {n} 的分片数量 {len(gather_list)} 不等于 tp_size {tp_size}"
                        )
                        continue
                    dim = attr_dict["row"]
                    full_param = torch.cat(gather_list, dim=dim)
                    tp_meta_data[n] = dim
                    param_name = ".".join(n.split(".")[1:]).replace("row_linear.", "")
                    model_state_dict[param_name] = full_param
                    total_size += full_param.numel()
                elif ("wte" in n or "tied_modules" in n) and parallel_vocab:
                    # 处理词嵌入和相关模块的参数
                    gather_list = [
                        tp_state_dicts[r]["model"][n] for r in range(tp_size)
                    ]
                    if len(gather_list) != tp_size:
                        logger.warning(
                            f"参数 {n} 的分片数量 {len(gather_list)} 不等于 tp_size {tp_size}"
                        )
                        continue
                    dim = 0
                    full_param = torch.cat(gather_list, dim=dim)
                    tp_meta_data[n] = dim
                    model_state_dict[n] = full_param
                    total_size += full_param.numel()
                else:
                    # 处理其他参数
                    param_name = ".".join(n.split(".")[1:])
                    model_state_dict[param_name] = p
                    total_size += p.numel()

        # 根据pipeline并行大小生成保存文件名
        if pp_size == 1:
            save_name = "model.safetensors"
        else:
            save_name = f"model-{pp_rank+1:05d}-of-{pp_size:06d}.safetensors"

        # 创建权重映射和元数据
        index_dict = {
            "metadata": {
                "total_size": {save_name: total_size},
                "format": "pt",
            },
            "tp_meta_data": tp_meta_data,
            "weight_map": {k: save_name for k in model_state_dict.keys()},
        }

        # 保存safetensors文件
        save_path_file = os.path.join(save_path, save_name)
        try:
            save_file(model_state_dict, save_path_file)
            logger.info(f"Successfully saved the weight file: {save_path_file}")
        except Exception as e:
            logger.error(f"Failed to save the weight file: {save_path_file}")
            raise e

        if pp_rank == 0:
            # 首次保存索引文件
            with open(index_file_path, "w") as f:
                json.dump(index_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully created the index file ‌: {index_file_path}")
        else:
            # 更新已存在的索引文件
            with open(index_file_path, "r") as f:
                existing_index = json.load(f)

            existing_index["metadata"]["total_size"].update(
                index_dict["metadata"]["total_size"]
            )
            existing_index["tp_meta_data"].update(index_dict["tp_meta_data"])
            existing_index["weight_map"].update(index_dict["weight_map"])

            with open(index_file_path, "w") as f:
                json.dump(existing_index, f, indent=2, ensure_ascii=False)
            logger.info("Successfully updated the index file")


def main() -> None:
    """
    Merak检查点转safetensors格式转换器.
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="将Merak格式的检查点文件转换为safetensors格式."
    )
    parser.add_argument(
        "--load_path", type=str, required=True, help="Merak检查点文件夹路径"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="保存safetensors格式文件的路径"
    )
    parser.add_argument(
        "--parallel_vocab", action="store_true", help="是否启用并行词汇表配置"
    )

    args = parser.parse_args()

    try:
        convert_safetensor_ckpt(args)
        logger.info("Checkpoint conversion completed.")
    except Exception as e:
        logger.error(f"An error occurred during the conversion process: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
