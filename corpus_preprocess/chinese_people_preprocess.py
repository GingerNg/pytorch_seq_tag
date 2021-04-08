import numpy as np
from collections import Counter
import os
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import codecs
import cfg
from utils import file_utils
import logging
from nlp_tools.tokenizers import WhitespaceTokenizer
import torch
from utils.model_utils import use_cuda, device


def _batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))  # ceil 向上取整
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - \
            1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]  # ???

        yield docs  # 　返回一个batch的数据


class DatasetProcesser(object):
    def __init__(self, bert_path):
        super().__init__()
        self.tokenizer = WhitespaceTokenizer(bert_path)

    def get_examples(self, data, label_encoder):
        label2id = label_encoder.label2id
        examples = []
        for dat in data:
        # for text, label in zip(data['text'], data['label']):
            # label
            ids = label2id(dat["tags"])
            token_ids = self.tokenizer.tokenize(dat["words"])
            # for sent_len, sent_words in sents_words:
            #     word_ids = vocab.word2id(sent_words)
            #     extword_ids = emb_vocab.extword2id(sent_words)
            #     # sent_len 句子长度：即句子中词个数
            #     doc.append([sent_len, word_ids, extword_ids])
            examples.append([ids, len(token_ids), token_ids])

        logging.info('Total %d docs.' % len(examples))
        return examples

    def data_iter(self, data, batch_size, shuffle=True, noise=1.0):
        batched_data = []
        if shuffle:
            np.random.shuffle(data)
            sorted_data = data
            # lengths = [example[1] for example in data]
            # noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
            # sorted_indices = np.argsort(noisy_lengths).tolist()
            # sorted_data = [data[i] for i in sorted_indices]
        else:
            sorted_data = data

        batch = list(_batch_slice(sorted_data, batch_size))
        batched_data.extend(batch)  # [[],[]]

        if shuffle:
            np.random.shuffle(batched_data)

        for batch in batched_data:
            yield batch

    def batch2tensor(self, batch_data):
        batch_size = len(batch_data)
        max_sent_len = 200
        doc_labels = []
        for doc_data in batch_data:
            if len(doc_data[0]) >= max_sent_len:
                doc_labels.extend(doc_data[0][0:max_sent_len])
            else:
                doc_labels.extend(doc_data[0] + [0]*(max_sent_len-len(doc_data[0])))
        batch_labels = torch.LongTensor(doc_labels)

        token_type_ids = [0] * max_sent_len
        batch_inputs1 = torch.zeros(
            (batch_size, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros(
            (batch_size, max_sent_len), dtype=torch.int64)
        # batch_labels = torch.zeros(
        #     (batch_size, max_sent_len), dtype=torch.int64)

        for b in range(batch_size):
            token_ids = batch_data[b][2]
            # ids = batch_data[b][0]
            for word_idx in range(min(max_sent_len, len(token_ids))):
                batch_inputs1[b, word_idx] = token_ids[word_idx]
                batch_inputs2[b, word_idx] = token_type_ids[word_idx]
                # batch_labels[b, word_idx] = ids[word_idx]

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_labels = batch_labels.to(device)
        print("batch_labels_shape:{}, batch_inputs1_shape:{}".format(batch_labels.shape, batch_inputs1.shape))
        return (batch_inputs1, batch_inputs2), batch_labels


class LabelEncoer():  # 标签编码
    def __init__(self):
        self.unk = 1
        self._id2label = PUNCTUATION_VOCABULARY
        self.target_names = PUNCTUATION_VOCABULARY
        # process label
        # label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政',
        #               5: '社会', 6: '教育', 7: '财经', 8: '家居', 9: '游戏',
        #               10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        # for label, name in label2name.items():
        #     self._id2label.append(label)
        #     self.target_names.append(name)

        def reverse(x): return dict(zip(x, range(len(x))))  # 词与id的映射
        self._label2id = reverse(self._id2label)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def label_size(self):
        return len(self.target_names)

    def label2name(self, xs):
        if isinstance(xs, list):
            return [self.target_names[x] for x in xs]
        return self.target_names[xs]