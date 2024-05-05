#!/bin/bash
# Data processing
python data/redial/extract_subkg.py --hop "${nhop}" --drop "${drop_rate}"
python data/redial/remove_entity.py

# Prompt pre-training
cp -r data/redial src/data/
python src/data/redial/process.py

code=1

until [ $code -eq 0 ]; do
  accelerate launch src/train_pre.py \
      --dataset redial \
      --tokenizer microsoft/DialoGPT-small \
      --model microsoft/DialoGPT-small \
      --text_tokenizer roberta-base \
      --text_encoder roberta-base \
      --num_train_epochs 5 \
      --gradient_accumulation_steps 1 \
      --per_device_train_batch_size 64 \
      --per_device_eval_batch_size 128 \
      --num_warmup_steps 1389 \
      --max_length 200 \
      --prompt_max_length 200 \
      --entity_max_length 32 \
      --learning_rate 5e-4 \
      --output_dir prompt-save
  code=$?
  if [ $code -ne 0 ]; then
    rm ${log_path}/*.log
  fi
done
