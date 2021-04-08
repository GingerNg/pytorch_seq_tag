# from data.common import SPACE, UNK, PAD, NUM, END, write_json
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
        self.tokenizer = WhitespaceTokenizer(bert_path, max_len=200)

    def get_examples(self, data, label_encoder):
        label2id = label_encoder.label2id
        examples = []
        for dat in data:
        # for text, label in zip(data['text'], data['label']):
            # label
            ids = label2id(dat["tags"])                        # 199
            token_ids = self.tokenizer.tokenize(dat["words"])  # 201
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


PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"
SPACE = "_SPACE"


def split_dataset(raw_path, train_path, dev_path, test_path):
    # file = '/data/dh/neural_sequence_labeling-master/data/raw/LREC/2014_corpus.txt'
    # file = peopledaily_corpus.raw_path
    with open(raw_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        processed_lines = []
        for line in tqdm(lines):

            line = re.sub(r'《', '', line)
            line = re.sub(r'》', '', line)
            line = re.sub(r'“', '', line)
            line = re.sub(r'”', '', line)
            line = re.sub(r'——', '', line)
            line = re.sub(r'‘', '', line)
            line = re.sub(r'：', '', line)
            line = re.sub(r'’', '', line)
            line = re.sub(r'\[', '', line)
            line = re.sub(r']', '', line)
            line = re.sub(r'{', '', line)
            line = re.sub(r'}', '', line)
            line = re.sub(r'【', '', line)
            line = re.sub(r'】', '', line)
            line = re.sub(r'（', '', line)
            line = re.sub(r'）', '', line)
            line = re.sub(r'—', '', line)
            line = re.sub(r'…', '', line)
            line = re.sub(r'/[a-z]+\d* *[\[\]]?', "", line)
            line = ''.join(char+' ' for char in line)
            line = re.sub(r'、', " ,COMMA", line)
            line = re.sub(r'！', " .PERIOD ", line)
            line = re.sub(r'；', " ,COMMA ", line)
            line = re.sub(r'，', " ,COMMA ", line)
            line = re.sub(r'。', " .PERIOD ", line)
            line = re.sub(r'？', " ?QUESTIONMARK ", line)
            line = line.replace('\n', ' ')
            processed_lines.append(line)
        train_set, dev_test_set = train_test_split(
            processed_lines, shuffle=True, test_size=0.2)
        print(len(dev_test_set))
        print(len(train_set))
        dev_set, test_set = train_test_split(
            dev_test_set, shuffle=True, test_size=0.5)
        print(len(dev_set))
        print(len(test_set))

    # write_txt(train_path, train_set)
    # write_txt(dev_path, dev_set)
    # write_txt(test_path, test_set)
    file_utils.write_data(train_set, train_path)
    file_utils.write_data(dev_set, dev_path)
    file_utils.write_data(test_set, test_path)
    print('done')


# pre-set number of records in different glove embeddings
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}

# Comma, period & question mark only:
PUNCTUATION_VOCABULARY = [SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"]
#PUNCTUATION_VOCABULARY = [SPACE, ",COMMA"]
PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD",
                       ":COLON": ",COMMA", ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
# punctuations that are not included in vocabulary nor mapping, must be added to CRAP_TOKENS
CRAP_TOKENS = {"<doc>", "<doc.>"}


def is_number(word):
    numbers = re.compile(r"\d")
    return len(numbers.sub("", word)) / len(word) < 0.6


def build_vocab_list(data_files, min_word_count, min_char_count, max_vocab_size):
    word_counter = Counter()
    char_counter = Counter()
    for file in data_files:
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                for word in line.lstrip().rstrip().split():
                    if word in CRAP_TOKENS or word in PUNCTUATION_VOCABULARY or word in PUNCTUATION_MAPPING:
                        continue
                    if is_number(word):
                        word_counter[NUM] += 1
                        for char in word:
                            char_counter[char] += 1
                        continue
                    word_counter[word] += 1
                    for char in word:
                        char_counter[char] += 1
    word_vocab = [word for word, count in word_counter.most_common() if count >= min_word_count and word != UNK and
                  word != NUM][:max_vocab_size]
    char_vocab = [char for char, count in char_counter.most_common(
    ) if count >= min_char_count and char != UNK]
    return word_vocab, char_vocab


def build_vocabulary(word_vocab, char_vocab):
    if NUM not in word_vocab:
        word_vocab.append(NUM)
    if END not in word_vocab:
        word_vocab.append(END)
    if UNK not in word_vocab:
        word_vocab.append(UNK)
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    if END not in char_vocab:
        char_vocab.append(END)
    if UNK not in char_vocab:
        char_vocab.append(UNK)
    if PAD not in char_vocab:
        char_vocab = [PAD] + char_vocab
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict


def load_glove_vocab(glove_path, glove_name):
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            vocab.add(line[0])
    return vocab


def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    scale = np.sqrt(3.0 / dim)
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])
    mask = np.zeros([len(word_dict)])
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=glove_sizes[glove_name], desc="Filter glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
            # since tokens in train sets are lowercase
            elif word.lower() in word_dict and mask[word_dict[word.lower()]] == 0:
                word = word.lower()
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_dataset(data_files, word_dict, char_dict, punct_dict, max_sequence_len):
    """
    data will consist of two sets of aligned sub-sequences (words and punctuations) of MAX_SEQUENCE_LEN tokens
    (actually punctuation sequence will be 1 element shorter).
    If a sentence is cut, then it will be added to next subsequence entirely
    (words before the cut belong to both sequences)
    """
    dataset = []
    current_words, current_chars, current_punctuations = [], [], []
    # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_eos_idx = 0
    last_token_was_punctuation = True  # skip first token if it's punctuation
    # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence
    skip_until_eos = False
    for file in data_files:
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                for token in line.split():
                    # First map oov punctuations to known punctuations
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]
                    if skip_until_eos:
                        if token in EOS_TOKENS:
                            skip_until_eos = False
                        continue
                    elif token in CRAP_TOKENS:
                        continue
                    elif token in punct_dict:
                        # if we encounter sequences like: "... !EXLAMATIONMARK .PERIOD ...",
                        # then we only use the first punctuation and skip the ones that follow
                        if last_token_was_punctuation:
                            continue
                        if token in EOS_TOKENS:
                            # no -1, because the token is not added yet
                            last_eos_idx = len(current_punctuations)
                        punctuation = punct_dict[token]
                        current_punctuations.append(token)
                        last_token_was_punctuation = True
                    else:
                        if not last_token_was_punctuation:
                            current_punctuations.append(SPACE)
                        chars = []
                        for c in token:
                            c = char_dict.get(c, char_dict[UNK])
                            chars.append(c)
                        if is_number(token):
                            token = NUM
                        word = word_dict.get(token, word_dict[UNK])
                        current_words.append(token)
                        current_chars.append(chars)
                        last_token_was_punctuation = False
                    # this also means, that last token was a word
                    if len(current_words) == max_sequence_len:
                        assert len(current_words) == len(current_punctuations) + 1, \
                            "#words: %d; #punctuations: %d" % (
                                len(current_words), len(current_punctuations))
                        # Sentence did not fit into subsequence - skip it
                        if last_eos_idx == 0:
                            skip_until_eos = True
                            current_words = []
                            current_chars = []
                            current_punctuations = []
                            # next sequence starts with a new sentence, so is preceded by eos which is punctuation
                            last_token_was_punctuation = True
                        else:
                            subsequence = {"words": current_words[:-1], "chars": current_chars[:-1],
                                           "tags": current_punctuations}
                            dataset.append(subsequence)
                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx + 1:]
                            current_chars = current_chars[last_eos_idx + 1:]
                            current_punctuations = current_punctuations[last_eos_idx + 1:]
                        last_eos_idx = 0  # sequence always starts with a new sentence
    return dataset


def process_data(config, train_path, dev_path):
    train_file = train_path
    dev_file = dev_path
    # train_file = os.path.join(config["raw_path"], "2014_train.txt")
    # dev_file = os.path.join(config["raw_path"], "2014_dev.txt")
    #ref_file = os.path.join(config["raw_path"], "2014_test.txt")

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    # build vocabulary
    word_vocab, char_vocab = build_vocab_list([train_file], config["min_word_count"], config["min_char_count"],
                                              config["max_vocab_size"])
    if not config["use_pretrained"]:
        word_dict, char_dict = build_vocabulary(word_vocab, char_vocab)
    else:
        #glove_path = config["glove_path"].format(config["glove_name"], config["emb_dim"])
        glove_path = config["glove_path"]
        glove_vocab = load_glove_vocab(glove_path, config["glove_name"])
        glove_vocab = glove_vocab & {word.lower() for word in glove_vocab}
        word_vocab = [word for word in word_vocab if word in glove_vocab]
        word_dict, char_dict = build_vocabulary(word_vocab, char_vocab)
        tmp_word_dict = word_dict.copy()
        del tmp_word_dict[UNK], tmp_word_dict[NUM], tmp_word_dict[END]
        vectors = filter_glove_emb(
            tmp_word_dict, glove_path, config["glove_name"], config["emb_dim"])
        np.savez_compressed(config["pretrained_emb"], embeddings=vectors)
    # create indices dataset
    punct_dict = dict([(punct, idx)
                       for idx, punct in enumerate(PUNCTUATION_VOCABULARY)])

    train_set = build_dataset(
        [train_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])
    dev_set = build_dataset([dev_file], word_dict,
                            char_dict, punct_dict, config["max_sequence_len"])
    #ref_set = build_dataset([ref_file], word_dict, char_dict, punct_dict, config["max_sequence_len"])

    # vocab = {"word_dict": word_dict,
    #          "char_dict": char_dict, "tag_dict": punct_dict}
    # write to file
    #write_json(config["vocab"], vocab)
    file_utils.write_json(config["train_set"], train_set)
    file_utils.write_json(config["dev_set"], dev_set)
    # write_json(config["ref_set"], ref_set)
    # write_json(config["asr_set"], asr_set)

