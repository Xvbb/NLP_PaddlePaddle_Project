import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SimCSE(nn.Layer):

    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.0,
                 scale=20,
                 output_emb_size=None):

        super().__init__()

        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                768, output_emb_size, weight_attr=weight_attr)

        self.margin = margin
        self.sacle = scale

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             with_pooler=True):

        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)

        if with_pooler == False:
            cls_embedding = sequence_output[:, 0, :]

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)

        return cls_embedding

    def cosine_sim(self,
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
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids,
            title_token_type_ids,
            title_position_ids,
            title_attention_mask,
            with_pooler=with_pooler)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

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
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True)

        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype())

        cosine_sim = cosine_sim - paddle.diag(margin_diag)
        cosine_sim *= self.sacle

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss
