accelerate launch train_rec.py --dataset redial_gen --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_rec 20 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_redial-resp_1e-3/final --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-20_redial_1e-4 --use_wandb --project crs-prompt-rec-final --name dialogpt_prompt-pre_prefix-20_redial_1e-4 --log_all
accelerate launch train_rec.py --dataset redial_gen --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_rec 10 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_redial-resp_1e-3/final --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-10_redial_1e-4 --use_wandb --project crs-prompt-rec-final --name dialogpt_prompt-pre_prefix-10_redial_1e-4 --log_all
accelerate launch train_rec.py --dataset redial_gen --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_rec 5 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_redial-resp_1e-3/final --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-5_redial_1e-4 --use_wandb --project crs-prompt-rec-final --name dialogpt_prompt-pre_prefix-5_redial_1e-4 --log_all
accelerate launch train_rec.py --dataset redial_gen --tokenizer ../utils/tokenizer/dialogpt-small --model microsoft/DialoGPT-small --text_tokenizer ../utils/tokenizer/roberta-base --text_encoder roberta-base --n_prefix_rec 3 --prompt_encoder /mnt/wangxiaolei/crs/prompt/dialogpt_redial-resp_1e-3/final --num_train_epochs 5 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --num_warmup_steps 530 --context_max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 1e-4 --output_dir /mnt/wangxiaolei/crs/prompt/dialogpt_prompt-pre_prefix-3_redial_1e-4 --use_wandb --project crs-prompt-rec-final --name dialogpt_prompt-pre_prefix-3_redial_1e-4 --log_all