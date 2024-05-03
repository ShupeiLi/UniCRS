#!/bin/bash
# Conversation Task Training and Inference
# train
cp -r data/inspired src/data/
python src/data/inspired/process_mask.py

code=1

until [ $code -eq 0 ]; do
  accelerate launch src/train_conv.py \
      --dataset inspired \
      --tokenizer microsoft/DialoGPT-small \
      --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base \
      --text_encoder roberta-base \
      --n_prefix_conv 20 \
      --prompt_encoder prompt-save/best \
      --num_train_epochs 10 \
      --gradient_accumulation_steps 1 \
      --ignore_pad_token_for_loss \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 16 \
      --num_warmup_steps 976 \
      --context_max_length 200 \
      --resp_max_length 183 \
      --prompt_max_length 200 \
      --entity_max_length 32 \
      --learning_rate 1e-4 \
      --output_dir conv-save
  code=$?
  if [ $code -ne 0 ]; then
    rm ${log_path}/*.log
  fi
done
