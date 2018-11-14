# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
import os

import numpy as np

from generator.corpus_reader import CorpusReader, str2id, padding, load_word_dict, save_word_dict
from generator.evaluate import Evaluate
from generator import config
from generator.seq2seq_attn_model import Seq2seqAttnModel


def data_generator(input_texts, target_texts, char2id, batch_size, maxlen=400):
    # 数据生成器
    while True:
        X, Y = [], []
        for i in range(len(input_texts)):
            X.append(str2id(input_texts[i], char2id, maxlen))
            Y.append(str2id(target_texts[i], char2id, maxlen))
            if len(X) == batch_size:
                X = np.array(padding(X, char2id))
                Y = np.array(padding(Y, char2id))
                yield [X, Y], None
                X, Y = [], []


def get_validation_data(input_texts, target_texts, char2id, maxlen=400):
    # 数据生成器
    X, Y = [], []
    for i in range(len(input_texts)):
        X.append(str2id(input_texts[i], char2id, maxlen))
        Y.append(str2id(target_texts[i], char2id, maxlen))
        X = np.array(padding(X, char2id))
        Y = np.array(padding(Y, char2id))
        return [X, Y], None


def train(train_path='',
          test_path='',
          save_vocab_path='',
          attn_model_path='',
          batch_size=64,
          epochs=100,
          maxlen=400,
          hidden_dim=128,
          min_count=5,
          dropout=0.2,
          use_gpu=False,
          sep='\t'):
    # load or save word dict
    if os.path.exists(save_vocab_path):
        token_2_id = load_word_dict(save_vocab_path)
        data_reader = CorpusReader(train_path=train_path, token_2_id=token_2_id, min_count=min_count, sep=sep)
    else:
        print('Training data...')
        data_reader = CorpusReader(train_path=train_path, min_count=min_count, sep=sep)
        token_2_id = data_reader.token_2_id
        save_word_dict(token_2_id, save_vocab_path)

    id_2_token = data_reader.id_2_token
    input_texts, target_texts = data_reader.build_dataset(train_path)
    test_input_texts, test_target_texts = data_reader.build_dataset(test_path)

    model = Seq2seqAttnModel(token_2_id,
                             attn_model_path=attn_model_path,
                             hidden_dim=hidden_dim,
                             use_gpu=use_gpu,
                             dropout=dropout).build_model()

    evaluator = Evaluate(model, attn_model_path, token_2_id, id_2_token, maxlen)
    model.fit_generator(data_generator(input_texts, target_texts, token_2_id, batch_size, maxlen),
                        steps_per_epoch=(len(input_texts) + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=get_validation_data(test_input_texts, test_target_texts, token_2_id, maxlen),
                        callbacks=[evaluator])


if __name__ == "__main__":
    train(train_path=config.train_path,
          test_path=config.test_path,
          save_vocab_path=config.save_vocab_path,
          attn_model_path=config.attn_model_path,
          batch_size=config.batch_size,
          epochs=config.epochs,
          maxlen=config.maxlen,
          hidden_dim=config.rnn_hidden_dim,
          min_count=config.min_count,
          dropout=config.dropout,
          use_gpu=config.use_gpu,
          sep=config.sep)
