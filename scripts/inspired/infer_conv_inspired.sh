#!/bin/bash
# Conversation Task Training and Inference
# infer
code=1

until [ $code -eq 0 ]; do
  accelerate launch src/infer_conv.py \
      --dataset inspired \
      --split train \
      --tokenizer microsoft/DialoGPT-small \
      --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base \
      --text_encoder roberta-base \
      --n_prefix_conv 20 \
      --prompt_encoder conv-save/best \
      --per_device_eval_batch_size 64 \
      --context_max_length 200 \
      --resp_max_length 183 \
      --prompt_max_length 200 \
      --entity_max_length 32
  code=$?
  if [ $code -ne 0 ]; then
    rm ${log_path}/*.log
  fi
done

mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/conv-infer-train.log"
mv "${log_path}/conv-infer-train.log" "${log_back_path}"

code=1

until [ $code -eq 0 ]; do
  accelerate launch src/infer_conv.py \
      --dataset inspired \
      --split valid \
      --tokenizer microsoft/DialoGPT-small \
      --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base \
      --text_encoder roberta-base \
      --n_prefix_conv 20 \
      --prompt_encoder conv-save/best \
      --per_device_eval_batch_size 64 \
      --context_max_length 200 \
      --resp_max_length 183 \
      --prompt_max_length 200 \
      --entity_max_length 32
  code=$?
  if [ $code -ne 0 ]; then
    rm ${log_path}/*.log
  fi
done

mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/conv-infer-valid.log"
mv "${log_path}/conv-infer-valid.log" "${log_back_path}"

code=1

until [ $code -eq 0 ]; do
  accelerate launch src/infer_conv.py \
      --dataset inspired \
      --split test \
      --tokenizer microsoft/DialoGPT-small \
      --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base \
      --text_encoder roberta-base \
      --n_prefix_conv 20 \
      --prompt_encoder conv-save/best \
      --per_device_eval_batch_size 64 \
      --context_max_length 200 \
      --resp_max_length 183 \
      --prompt_max_length 200 \
      --entity_max_length 32
  code=$?
  if [ $code -ne 0 ]; then
    rm ${log_path}/*.log
  fi
done

mv "${log_path}/$(ls "./log" | grep ".*\.log")" "${log_path}/conv-infer-test.log"
mv "${log_path}/conv-infer-test.log" "${log_back_path}"
