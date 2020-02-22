import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *


class LSyE(fr.LSSE):
    name = "LSyE"
    result_path = file_tool.connect_path('result', name)

    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.name = LSyE.name
        self.result_path = LSyE.result_path

    @classmethod
    def framework_name(cls):
        return cls.name

    def create_arg_dict(self):
        arg_dict = {
            # 'sgd_momentum': 0.4,
            'semantic_compare_func': 'l2',
            'concatenate_input_for_gcn_hidden': True,
            'fully_scales': [768, 150, 1],
            'position_encoding': True,
            'fully_regular': 1e-4,
            'gcn_regular': 1e-4,
            'bert_regular': 1e-4,
            'gcn_layer': 2,
            'group_layer_limit_flag': False,
            'group_layer_limit_list': [2, 3, 4, 5, 6],
            'gcn_gate_flag': True,
            'gcn_norm_item': 0.5,
            'gcn_self_loop_flag': True,
            'gcn_hidden_dim': 768,
            'bert_hidden_dim': 768,
            'dtype': torch.float32,
        }
        return arg_dict

    def forward(self, *input_data, **kwargs):
        if len(kwargs) == 5: # common run or visualization
            data_batch = kwargs
            sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
            segment_ids = data_batch['segment_ids']
            sep_index = data_batch['sep_index']
            adj_matrix1s = data_batch['adj_matrix1s']
            adj_matrix2s = data_batch['adj_matrix2s']
        else:
            sentence_pair_tokens, segment_ids, sep_index, adj_matrix1s, adj_matrix2s = input_data

        _, sentence1_reps, sentence2_reps = self.bert(sentence_pair_tokens, segment_ids, sep_index)

        def get_position_es(shape):
            position_encodings = general_tool.get_global_position_encodings(length=self.arg_dict['max_sentence_length'],
                                                                            dimension=self.arg_dict['bert_hidden_dim'])
            position_encodings = position_encodings[:shape[1]]
            position_encodings = torch.tensor(position_encodings, dtype=self.data_type,
                                              device=self.device).expand([shape[0], -1, -1])
            return position_encodings

        if self.arg_dict['position_encoding']:
            shape1 = sentence1_reps.size()
            position_es1 = get_position_es(shape1)
            shape2 = sentence2_reps.size()
            position_es2 = get_position_es(shape2)
            sentence1_reps += position_es1
            sentence2_reps += position_es2

        # star_time = time.time()
        gcn_out1 = self.gcn(sentence1_reps, adj_matrix1s)
        gcn_out2 = self.gcn(sentence2_reps, adj_matrix2s)
        if self.arg_dict['concatenate_input_for_gcn_hidden']:
            gcn_out1 = torch.cat([gcn_out1, sentence1_reps], dim=2)
            gcn_out2 = torch.cat([gcn_out2, sentence2_reps], dim=2)
        result = self.semantic_layer(gcn_out1, gcn_out2)

        result = self.fully_connection(result)
        return result


