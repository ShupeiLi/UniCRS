#!/bin/bash
# Change nhop, drop_rate, dataset
nhop=1
drop_rate=1.0
dataset="redial"

log_path="$(pwd)/log"
log_back_path="$(pwd)/log-backup"

save_path="/root/autodl-fs/unicrs/${dataset}-dbpedia"

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
mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/rec-train.log"
mv "${log_path}/rec-train.log" "${log_back_path}"
mv "${log_path}/$(ls "./log" | grep ".*\.jsonl")" "${log_back_path}"

# backup
mkdir -p "${save_path}/${drop_rate}-hop/"
mv "save/${dataset}" log-backup
cp -R log-backup "${save_path}/${drop_rate}-hop/"
mkdir -p model
mv prompt-save model
mv conv-save model
mv rec-save model
zip -r "${dataset}-dbpedia-${drop_rate}hop-model.zip" model
mv "${dataset}-dbpedia-${drop_rate}hop-model.zip" "${save_path}/${drop_rate}-hop/"

# clean up
sh scripts/clean.sh
