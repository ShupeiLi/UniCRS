#!/bin/bash
# Change n_hop, drop_rate, dataset
n_hop=2
drop_rate=1.0
dataset="redial"

log_path="$(pwd)/log"
log_back_path="$(pwd)/log-backup"

mkdir -p "${log_back_path}"

# Prompt pre-training
. "scripts/${dataset}/train_pre_${dataset}.sh"
mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/prompt.log"
mv "${log_path}/prompt.log" "${log_back_path}"

# Conversation training
. "scripts/${dataset}/train_conv_${dataset}.sh"
mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/conv-train.log"
mv "${log_path}/conv-train.log" "${log_back_path}"

# Conversation inference
. "scripts/${dataset}/infer_conv_${dataset}.sh"

# Recommendation task
. "scripts/${dataset}/train_rec_${dataset}.sh"
mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/conv-train.log"
mv "${log_path}/conv-train.log" "${log_back_path}"
mv "${log_path}/$(ls "./log" | grep ".*\.jsonl")" "${log_back_path}"
