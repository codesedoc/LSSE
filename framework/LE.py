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
            'fully_scales': [768, 150, 2],
            # 'fully_regular': 1e-4,
            # 'bert_regular': 1e-4,
            'bert_hidden_dim': 768,
            'pad_on_right': True,
            'sentence_max_len_for_bert': 128,
            'dtype': torch.float32,
        }
        return arg_dict

    def create_models(self):
        self.bert = BertBase()
        self.semantic_layer = SemanticLayer(self.arg_dict)
        self.fully_connection = FullyConnection(self.arg_dict)

    def forward(self, *input_data, **kwargs):
        if len(kwargs) > 0:  # common run or visualization
            data_batch = kwargs
            input_ids_batch = data_batch['input_ids_batch']
            token_type_ids_batch = data_batch['token_type_ids_batch']
            attention_mask_batch = data_batch['attention_mask_batch']
            sep_index_batch = data_batch['sep_index_batch']
            sent1_len_batch = data_batch['sent1_len_batch']
            sent2_len_batch = data_batch['sent2_len_batch']
            labels = data_batch['labels']

        else:
            input_ids_batch, token_type_ids_batch, attention_mask_batch, sep_index_batch, sent1_len_batch, \
            sent2_len_batch, labels = input_data

        last_hidden_states_batch, _ = self.bert(input_ids_batch, token_type_ids_batch, attention_mask_batch)

        sent1_states_batch = []
        sent2_states_batch = []
        for i, hidden_states in enumerate(last_hidden_states_batch):
            sent1_states = hidden_states[1:sep_index_batch[i]]
            sent2_states = hidden_states[sep_index_batch[i] + 1: sep_index_batch[i] + 1 + sent2_len_batch[i]]
            if len(sent1_states) != sent1_len_batch[i] or len(sent2_states) != sent2_len_batch[i]:
                raise ValueError
            if len(sent1_states) + len(sent2_states) + 3 != attention_mask_batch[i].sum():
                raise ValueError
            sent1_states = data_tool.padding_tensor(sent1_states, self.arg_dict['max_sentence_length'],
                                                    align_dir='left', dim=0)
            sent2_states = data_tool.padding_tensor(sent2_states, self.arg_dict['max_sentence_length'],
                                                    align_dir='left', dim=0)
            sent1_states_batch.append(sent1_states)
            sent2_states_batch.append(sent2_states)

        sent1_states_batch = torch.stack(sent1_states_batch, dim=0)
        sent2_states_batch = torch.stack(sent2_states_batch, dim=0)

        result = self.semantic_layer(sent1_states_batch, sent2_states_batch)

        result = self.fully_connection(result)

        loss = torch.nn.CrossEntropyLoss()(result.view(-1, 2), labels.view(-1))
        predicts = np.array(result.detach().cpu().numpy()).argmax(axis=1)

        return loss, predicts



