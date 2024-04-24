#!/bin/bash
# Data processing
python data/inspired/extract_subkg.py
python data/inspired/remove_entity.py

# Prompt pre-training
cp -r data/inspired src/data/
python src/data/inspired/process.py
accelerate launch src/train_pre.py \
    --dataset inspired \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --num_warmup_steps 168 \
    --max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 6e-4 \
    --output_dir prompt-save \
