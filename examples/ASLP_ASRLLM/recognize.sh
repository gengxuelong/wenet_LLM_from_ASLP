#!/bin/bash

. ./path.sh || exit 1;


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
stage=5 # start from 0 if you need to start from data_list preparation
stop_stage=5

decode_checkpoint=/home/work_nfs15/asr_data/ckpt/asrllm/13000hour_model/epoch_4.pt
decode_config_path=./conf/train_ASLP_ASRLLM.yaml
output_dir=./exp
mkdir -p $output_dir

gpu_id=7
decode_modes="ASRLLM_decode"
test_data_dir="/home/work_nfs15/asr_data/data/asr_test_sets"
test_sets="aishell aishell2"
. tools/parse_options.sh || exit 1;


set -e
set -u
set -o pipefail


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  for test_set in $test_sets; do
  {
    echo "test this dataset: $test_set"
    test_dir=$output_dir/test/${test_set}
    mkdir -p $test_dir
    export CUDA_VISIBLE_DEVICES=$gpu_id
    python wenet/bin/recognize.py --gpu $gpu_id \
      --modes $decode_modes \
      --config $decode_config_path \
      --data_type "raw" \
      --test_data $test_data_dir/$test_set/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --result_dir $test_dir
    echo "$test_set has been decoded!"
    python tools/compute-wer.py --char=1 --v=1 \
      $test_data_dir/$test_set/text $test_dir/text > $test_dir/wer
  }
  done
  wait
fi


