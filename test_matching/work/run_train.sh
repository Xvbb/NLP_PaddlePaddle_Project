unset CUDA_VISIBLE_DEVICES
python train.py \
	--device gpu \
	--save_dir ./checkpoint/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 1 \
	--save_steps 100 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--dropout 0.3 \
    --language "Chinese" \
    --pretrained_model "ernie" \
	--train_set_file "./data/data53912/train.tsv" \
	--test_set_file "./data/data53912/dev.tsv"
