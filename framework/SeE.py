import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *


class SeE(fr.LSeE):
    name = "SeE"
    result_path = file_tool.connect_path('result', name)

    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.name = SeE.name
        self.result_path = SeE.result_path

    @classmethod
    def framework_name(cls):
        return cls.name

    def create_arg_dict(self):
        arg_dict = {
            # 'sgd_momentum': 0.4,
            'fully_scales': [768, 150, 1],
            'fully_regular': 1e-4,
            'bert_regular': 1e-4,
            'bert_hidden_dim': 768,
            'dtype': torch.float32,
        }
        return arg_dict

    def create_models(self):
        self.bert = ALBertBase()
        self.fully_connection = BertFineTuneConnection(self.arg_dict)

    def update_arg_dict(self, arg_dict):
        self.arg_dict.update(arg_dict)

        if self.arg_dict['repeat_train']:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            model_dir = file_tool.connect_path(self.result_path, 'train',
                                               'bs:{}-lr:{}-fr:{}-br:{}'.
                                               format(self.arg_dict['batch_size'], self.arg_dict['learn_rate'],
                                                      self.arg_dict['fully_regular'], self.arg_dict['bert_regular'], ),
                                               time_str)

        else:
            model_dir = file_tool.connect_path(self.result_path, 'test')

        file_tool.makedir(model_dir)
        if not file_tool.check_dir(model_dir):
            raise RuntimeError
        self.arg_dict['model_path'] = model_dir

    def forward(self, *input_data, **kwargs):
        if len(kwargs) == 3: # common run or visualization
            data_batch = kwargs
            sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
            segment_ids = data_batch['segment_ids']
            sep_index = data_batch['sep_index']
        else:
            sentence_pair_tokens, segment_ids, sep_index = input_data

        sentence_pair_reps, _, _ = self.bert(sentence_pair_tokens, segment_ids, sep_index)

        # star_time = time.time()

        result = self.fully_connection(sentence_pair_reps)
        return result


