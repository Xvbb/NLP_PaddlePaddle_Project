python evaluate.py \
        --device gpu \
        --save_dir './checkpoint/best_model/' \
        --params_path './checkpoint/best_model/model_state.pdparams' \
        --batch_size 64 \
        --max_seq_length 64 \
        --test_set_file './data/data53912/test.tsv'
