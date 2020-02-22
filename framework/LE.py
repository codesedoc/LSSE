import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *


class LE(fr.LSeE):
    name = "LE"
    result_path = file_tool.connect_path('result', name)

    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.name = LE.name
        self.result_path = LE.result_path

    @classmethod
    def framework_name(cls):
        return cls.name

    def create_arg_dict(self):
        arg_dict = {
            'semantic_compare_func': 'l2',
            'fully_scales': [768, 150, 1],
            'fully_regular': 1e-4,
            'bert_regular': 1e-4,
            'bert_hidden_dim': 768,
            'dtype': torch.float32,
        }
        return arg_dict

    def forward(self, *input_data, **kwargs):
        if len(kwargs) == 3:  # common run or visualization
            data_batch = kwargs
            sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
            segment_ids = data_batch['segment_ids']
            sep_index = data_batch['sep_index']
        else:
            sentence_pair_tokens, segment_ids, sep_index = input_data

        sentence_pair_reps, sentence1_reps, sentence2_reps = self.bert(sentence_pair_tokens, segment_ids, sep_index)

        # star_time = time.time()
        result = self.semantic_layer(sentence1_reps, sentence2_reps)

        result = self.fully_connection(result)
        return result



