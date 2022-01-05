from functools import partial
import argparse
import os
import random

import numpy as np
import paddle

from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

from model import ESimCSE
from data import read_text_pair, convert_example, create_dataloader

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoints/best_model/', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=0, type=int, help="Output_embedding_size, 0 means use hidden_size as output embedding size.")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
# parser.add_argument("--text_pair_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument("--test_set_file", type=str, default='./data/test.tsv', required=True,  help="The full path of test_set_file.")
parser.add_argument("--margin", default=0.0, type=float, help="Margin beteween pos_sample and neg_samples.")
parser.add_argument("--scale", default=20, type=int, help="Scale for pair-wise margin_rank_loss.")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for pretrained model encoder.")
parser.add_argument("--infer_with_fc_pooler", action='store_true', help="Whether use fc layer after cls embedding or not for when infer.")
args = parser.parse_args()

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def do_evaluate(model, data_loader, with_pooler=False):
    model.eval()
    sims = []
    for batch in data_loader:
        query_input_ids, query_token_type_ids, query_attention_mask,title_input_ids, title_token_type_ids, title_attention_mask = batch
        batch_cosine_sim = model.get_cosine_sim(
            query_input_ids=query_input_ids,
            title_input_ids=title_input_ids,
            query_token_type_ids=query_token_type_ids,
            title_token_type_ids=title_token_type_ids,
            query_attention_mask=query_attention_mask,
            title_attention_mask=title_attention_mask,
            with_pooler=with_pooler).numpy()

        sims.append(batch_cosine_sim)

    sims = np.concatenate(sims, axis=0)

    return sims


def do_train():
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    test_ds = load_dataset(
        read_text_pair, data_path=args.test_set_file, is_test=True, lazy=False)

    
    # pretrained_model = AutoModel.from_pretrained(args.save_dir)
    # tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    pretrained_model = AutoModel.from_pretrained('ernie-1.0')
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')
    
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    test_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # query_attetntion_mask
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # title_segement
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)  # title_attention_mask
    ): [data for data in fn(samples)]

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=test_batchify_fn,
        trans_fn=trans_func)

    model = ESimCSE(
        pretrained_model,
        margin=args.margin,
        scale=args.scale,
        output_emb_size=args.output_emb_size
    )

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    sims = do_evaluate(
        model, test_data_loader, args.infer_with_fc_pooler)

    for index, cosine in enumerate(sims):
        if index < 10:
            print("text_a: {}, text_b: {}, similarity: {}".format(
                test_ds.data[index]['text_a'], test_ds.data[index]['text_b'], cosine))


if __name__ == "__main__":
    do_train()
