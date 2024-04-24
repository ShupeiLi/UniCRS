#!/bin/bash
# Recommendation Task
cp -r src/data/inspired/. src/data/inspired_gen/
python src/data/inspired_gen/merge.py --gen_file_prefix conv-save
accelerate launch src/train_rec.py \
    --dataset inspired_gen \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --n_prefix_rec 10 \
    --prompt_encoder prompt-save/best \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 33 \
    --context_max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 1e-4 \
    --output_dir rec-save \
