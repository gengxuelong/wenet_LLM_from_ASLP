#!/bin/bash

. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

train_config=conf/train_ASLP_ASRLLM.yaml
dir=./exp/ASLP_ASRLLM
mkdir -p $dir

checkpoint=

# raw or shard 更加推荐shard类型
data_type=shard

# 训练使用到的数据， 文件内容为若干tar文件path的集合
train_data=./data_list/gxl_all_new_wenetspeech_fix.list
cv_data=data_list/aishell1_dev_data.list

num_workers=6  # 数据加载的进程数
prefetch=400


train_engine="torch_ddp" # deepspeed or torch_ddp
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail



mkdir -p $dir
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
dist_backend="nccl"

if [ ${train_engine} == "deepspeed" ]; then
  echo "$0: using deepspeed"
else
  echo "$0: using torch ddp"
fi

echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
torchrun  --standalone  --nnodes=$num_nodes --nproc_per_node=$num_gpus \
         --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
  wenet/bin/train.py \
    --train_engine ${train_engine} \
    --config $train_config \
    --data_type  $data_type \
    --train_data $train_data \
    --cv_data $cv_data \
    ${checkpoint:+--checkpoint $checkpoint} \
    --model_dir $dir \
    --tensorboard_dir $dir/tensorboard \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --deepspeed_config ${deepspeed_config} \
    --deepspeed.save_states ${deepspeed_save_states} \
    --timeout 1200 \
    --use_amp





