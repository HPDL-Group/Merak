#! /bin/bash
hostname
GPUS_PER_NODE=4

# Change for multinode config
master=`scontrol show hostname $SLURM_NODELIST | head -n 1`

MASTER_ADDR=$master
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export NCCL_SOCKET_IFNAME=ib0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo $DISTRIBUTED_ARGS
python -um torch.distributed.launch $DISTRIBUTED_ARGS \
	convert_megatron_gpt2_checkpoint.py --path_to_checkpoint ./checkpoint_org --path_to_output_checkpoint ./checkpoint_final --dp 1 --tp 4 --pp 4 \
	--print-checkpoint-structure --overlap_level 3 \


# mpi, cpu
#hostname
#GPUS_PER_NODE=4
#
## Change for multinode config
#master=`scontrol show hostname $SLURM_NODELIST | head -n 1`
#
#export MASTER_ADDR=$master
#export MASTER_PORT=6000
#export NNODES=$SLURM_NNODES
##export RANK=$SLURM_PROCID
#export RANK=$OMPI_COMM_WORLD_RANK
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
#export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
#
#export NCCL_SOCKET_IFNAME=ib0
#
#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
#
#echo $DISTRIBUTED_ARGS
##python -um torch.distributed.launch $DISTRIBUTED_ARGS \
#python -u \
#        convert_megatron_gpt2_checkpoint.py --path_to_checkpoint ./checkpoint_org --path_to_output_checkpoint ./checkpoint_final --dp 1 --tp 4 --pp 4 \
#        --print-checkpoint-structure --overlap_level 3 \
#        --distributed_backend "mpi" --local_rank $LOCAL_RANK
