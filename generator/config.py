# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

data_dir = './data'
# raw_src_path = data_dir + '/src_sample.txt'
# raw_tgt_path = data_dir + '/tgt_sample.txt'

# Training data path.
train_path = os.path.join(data_dir, 'train_sample.txt')
# Validation data path.
test_path = os.path.join(data_dir, 'test_sample.txt')

output_dir = './output'
# seq2seq_attn_train config
save_vocab_path = os.path.join(output_dir, 'vocab.txt')
attn_model_path = os.path.join(output_dir, 'attn_model.weight')

# config
batch_size = 32
epochs = 100
rnn_hidden_dim = 128
maxlen = 400
min_count = 50
dropout = 0.0
use_gpu = False
sep = '\t'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
