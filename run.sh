export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node 7 --master_port 29501 \
    run_cls.py \
    --train_data_file data/train.csv \
    --eval_data_file data/eval.csv \
    --output_dir data/cps/bert_base \
    --output_best_dir data/cps/bert_base/best \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --save_total_limit 50 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --max_len 256 \
    --patience 2000 \
    --seed 42 \
    --logging_steps 200 \
    --save_steps 200 \
    --evaluate_during_training True \
    --model_name_or_path bert-base-chinese \
    --logging_dir logs/bert_base
