# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Corpus for model
from codecs import open

from generator.utils.io_utils import get_logger
from generator.reader import Reader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN, PAD_ID, EOS_ID, GO_ID

logger = get_logger(__name__)


def save_word_dict(dict_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("%s\t%d\n" % (k, v))


def load_word_dict(save_path):
    dict_data = dict()
    with open(save_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.split()
            try:
                dict_data[items[0]] = int(items[1])
            except IndexError:
                logger.error('error', line)
    return dict_data


class CorpusReader(Reader):
    """
    Read CGED data set
    """
    UNKNOWN_TOKEN = 'UNK'

    def __init__(self, train_path=None, token_2_id=None):
        super(CorpusReader, self).__init__(
            train_path=train_path, token_2_id=token_2_id,
            special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN, CorpusReader.UNKNOWN_TOKEN])
        self.UNKNOWN_ID = self.token_2_id[CorpusReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                line = line.lower().strip()
                if not line:
                    break
                if '\t' not in line: continue
                source, target = line.split('\t')
                yield source.split(), target.split()

    def unknown_token(self):
        return CorpusReader.UNKNOWN_TOKEN

    def read_tokens(self, path, is_infer=False):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.lower().strip().split()

    @staticmethod
    def read_vocab(input_texts, min_count=0):
        vocab = [PAD_TOKEN, EOS_TOKEN, GO_TOKEN]
        vocab_dict = dict()
        for line in input_texts:
            for char in line:
                if char not in vocab_dict:
                    vocab_dict[char] = 1
                else:
                    vocab_dict[char] += 1
        vocab_use = [i for i, j in vocab_dict.items() if j > min_count]
        vocab.extend(vocab_use)
        # vocab_sort = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted(vocab)


def str2id(s, char2id, maxlen):
    # 文字转整数id
    return [char2id.get(c, 1) for c in s[:maxlen]]


def padding(x, char2id):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [char2id[PAD_TOKEN]] * (ml - len(i)) for i in x]


def id2str(ids, id2char):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])
