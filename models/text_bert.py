from nlp_tools.tokenizers import WhitespaceTokenizer
from torch import nn
import logging
from transformers import BertModel
from utils.model_utils import use_cuda, device
import numpy as np
from cfg import bert_path
import torch.nn.functional as F

# build word encoder
dropout = 0.15


class BertSoftmaxModel(nn.Module):
    def __init__(self):
        super(BertSoftmaxModel, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)

        self.dense = nn.Linear(768, 84)
        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, input_ids, token_type_ids):
        # input_ids: sen_num x bert_len
        # token_type_ids: sen_num  x bert_len

        # sen_num x bert_len x 256, sen_num x 256
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids)

        score = F.softmax(sequence_output, dim=-1)  # dim=-1： 对最后一维进行softmax

        return score
        # if self.pooled:
        #     reps = pooled_output
        # else:
        #     reps = sequence_output[:, 0, :]  # sen_num x 256

        # if self.training:
        #     reps = self.dropout(reps)

        # logits = tf.layers.dense(embedding, units=num_labels, use_bias=True)
        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)

        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # per_example_loss = - \
        #     tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # loss 交叉熵损失函数
        # loss = tf.reduce_mean(per_example_loss)

