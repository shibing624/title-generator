# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
import os

from generator import config
from generator.corpus_reader import load_word_dict
from generator.evaluate import gen_target
from generator.seq2seq_attn_model import Seq2seqAttnModel


class Inference(object):
    def __init__(self, save_vocab_path='', attn_model_path='', maxlen=400):
        if os.path.exists(save_vocab_path):
            self.char2id = load_word_dict(save_vocab_path)
            self.id2char = {int(j): i for i, j in self.char2id.items()}
            self.chars = set([i for i in self.char2id.keys()])
        else:
            print('not exist vocab path')
        seq2seq_attn_model = Seq2seqAttnModel(self.chars, attn_model_path=attn_model_path)
        self.model = seq2seq_attn_model.build_model()
        self.maxlen = maxlen

    def infer(self, sentence):
        return gen_target(sentence, self.model, self.char2id, self.id2char, self.maxlen, topk=3)


if __name__ == "__main__":
    inputs = [
        "Field &amp; Main Bank purchased a new position in PowerShares Fin . Preferred Port . ( NYSEARCA : PGF )  "
        "in the fourth quarter , according to its most recent disclosure with the SEC . The institutional investor "
        "purchased 22,550 shares of the exchange traded fund 's stock , valued at approximately $ 425,000 . "
        "Other large investors also recently modified their holdings of the company . Cedar Hill Associates LLC "
        "acquired a new stake in shares of PowerShares Fin . Preferred Port .",
    ]
    inference = Inference(save_vocab_path=config.save_vocab_path,
                          attn_model_path=config.attn_model_path,
                          maxlen=config.maxlen)
    for i in inputs:
        target = inference.infer(i)
        print('input:' + i)
        print('output:' + target)
    while True:
        sent = input('input:')
        print("output:" + inference.infer(sent))
