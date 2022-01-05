import random

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ESimCSE(nn.Layer):
    # 网络组网
    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.0,
                 scale=20,
                 pooling="avg",
                 queue_size=160,
                 dup_rate=0.32,
                 output_emb_size=None):
        assert pooling in ["cls", "cls_before_pooler", "avg", "avg_top2",
                           "avg_first_last"], "unrecognized pooling type %s" % self.pooling
        super().__init__()

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.pooling = pooling
        self.queue_size = queue_size
        self.queue = []
        self.dup_rate = dup_rate
        self.gamma = 0.995
        self.moco = pretrained_model
        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                768, output_emb_size, weight_attr=weight_attr)

        self.margin = margin
        self.scale = scale

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             dropout=True,
                             with_pooler=True):

        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.pretrained_model(
            input_ids, token_type_ids)

        if with_pooler == False:
            cls_embedding = sequence_output[:, 0, :]

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)

        if dropout is True:
            cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def get_cosine_sim(self,
                       query_input_ids,
                       title_input_ids,
                       query_token_type_ids=None,
                       query_position_ids=None,
                       query_attention_mask=None,
                       title_token_type_ids=None,
                       title_position_ids=None,
                       title_attention_mask=None,
                       with_pooler=True):

        query_cls_embedding = self.get_pooled_embedding(
            input_ids=query_input_ids, token_type_ids=query_token_type_ids)

        # word_repetition
        rep_dict = self.word_repetition(
            title_input_ids, title_attention_mask, title_token_type_ids)
        title_input_ids, title_attention_mask, title_token_type_ids = rep_dict[
            'input_ids'], rep_dict['attention_mask'], rep_dict['token_type_ids']

        title_cls_embedding = self.get_pooled_embedding(
            input_ids=title_input_ids, token_type_ids=title_token_type_ids, attention_mask=title_attention_mask)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def word_repetition(self, input_ids, attention_mask, token_type_ids):
        bsz = len(input_ids)
        seq_len = len(input_ids[0])
        rep_seq_len = seq_len   
        # print(input_ids)
        assert attention_mask is not None
        repetition_inputs_ids, repetition_attention_mask, repetition_token_type_ids = [], [], []

        for bsz_id in range(bsz):
            sample_mask = paddle.tolist(attention_mask[bsz_id])
            cur_len = sum(sample_mask)
            cur_input_id = paddle.tolist(input_ids[bsz_id])
            cur_token_type_id = paddle.tolist(token_type_ids)
            dup_len = random.randint(a=0, b=max(
                2, int(self.dup_rate * cur_len)))
            dup_word_ids = random.sample(list(range(1, cur_len)), k=dup_len)

            r_input_id = []
            r_attetion_mask = []
            r_token_type_ids = []
            for index, word_id in enumerate(cur_input_id):
                if index in dup_word_ids:
                    r_input_id.append(word_id)
                    r_attetion_mask.append(sample_mask[index])
                    r_token_type_ids.append(cur_token_type_id[bsz_id][index])

                r_input_id.append(word_id)
                r_attetion_mask.append(sample_mask[index])
                r_token_type_ids.append(cur_token_type_id[bsz_id][index])

            after_dup_len = len(r_input_id)
            repetition_inputs_ids.append(r_input_id)
            repetition_attention_mask.append(r_attetion_mask)
            repetition_token_type_ids.append(r_token_type_ids)

            assert after_dup_len == dup_len + seq_len
            if after_dup_len > rep_seq_len:
                rep_seq_len = after_dup_len

        for i in range(bsz):
            after_dup_len = len(repetition_inputs_ids[i])
            pad_len = rep_seq_len - after_dup_len
            # padding
            repetition_inputs_ids[i] += [0] * pad_len
            repetition_attention_mask[i] += [0] * pad_len
            repetition_token_type_ids[i] += [0] * pad_len

        # for _ in repetition_token_type_ids:
        #     assert len(_) == len(repetition_inputs_ids[0])
        repetition_inputs_ids = paddle.to_tensor(repetition_inputs_ids)
        repetition_attention_mask = paddle.to_tensor(repetition_attention_mask)
        repetition_token_type_ids = paddle.to_tensor(repetition_token_type_ids)

        return {'input_ids': repetition_inputs_ids, 'attention_mask': repetition_attention_mask, 'token_type_ids': repetition_token_type_ids}

    def momentum_contrast():

        return

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            input_ids=query_input_ids, token_type_ids=query_token_type_ids)

        # word_repetition
        rep_dict = self.word_repetition(
            title_input_ids, title_attention_mask, title_token_type_ids)
        title_input_ids, title_attention_mask, title_token_type_ids = rep_dict[
            'input_ids'], rep_dict['attention_mask'], rep_dict['token_type_ids']

        title_cls_embedding = self.get_pooled_embedding(
            input_ids=title_input_ids, token_type_ids=title_token_type_ids, attention_mask=title_attention_mask)

        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True) * self.scale

        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype())

        cosine_sim = cosine_sim - paddle.diag(margin_diag)
        cosine_sim *= self.scale
        

        negative_samples = None
        if len(self.queue) > 0:
            negative_samples = paddle.concat(
                self.queue[:self.queue_size], axis=0)

        batch_size = cosine_sim.shape[0]
        if len(self.queue) + batch_size >= self.queue_size:
            del self.queue[:batch_size]
        # print('此时q队列中的样本数: ', len(self.queue))

        with paddle.no_grad():
            self.queue.append(self.get_pooled_embedding(
                query_input_ids, token_type_ids=query_token_type_ids, dropout=False))

        if negative_samples is not None:
            batch_size += negative_samples.shape[0]   
            cos_sim_with_neg = paddle.matmul(
                query_cls_embedding, negative_samples, transpose_y=True) * self.scale   
            cosine_sim = paddle.concat(
                [cosine_sim, cos_sim_with_neg], axis=1)

        for encoder_param, moco_encoder_param in zip(self.pretrained_model.parameters(), self.moco.parameters()):
            moco_encoder_param = self.gamma * moco_encoder_param + \
                (1. - self.gamma) * encoder_param


        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss