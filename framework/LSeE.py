import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *


class LSeE(fr.Framework):
    name = "LSeE"
    result_path = file_tool.connect_path('result', name)

    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.name = LSeE.name
        self.result_path = LSeE.result_path

    @classmethod
    def framework_name(cls):
        return cls.name

    def create_arg_dict(self):
        arg_dict = {
            'semantic_compare_func': 'l2',
            'fully_scales': [768 * 2, 150, 1],
            'fully_regular': 1e-4,
            'bert_regular': 1e-4,
            'bert_hidden_dim': 768,
            'dtype': torch.float32,
        }
        return arg_dict

    def update_arg_dict(self, arg_dict):
        super().update_arg_dict(arg_dict)

        if self.arg_dict['repeat_train']:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            model_dir = file_tool.connect_path(self.result_path, 'train',
                                               'bs:{}-lr:{}-fr:{}-br:{}-com_fun:{}'.
                                               format(self.arg_dict['batch_size'], self.arg_dict['learn_rate'],
                                                      self.arg_dict['fully_regular'], self.arg_dict['bert_regular'],
                                                      self.arg_dict['semantic_compare_func']), time_str)

        else:
            model_dir = file_tool.connect_path(self.result_path, 'test')

        file_tool.makedir(model_dir)
        if not file_tool.check_dir(model_dir):
            raise RuntimeError
        self.arg_dict['model_path'] = model_dir

    def create_models(self):
        self.bert = ALBertBase()
        self.semantic_layer = SemanticLayer(self.arg_dict)
        self.fully_connection = FullyConnection(self.arg_dict)

    def deal_with_example_batch(self, example_ids, example_dict):
        examples = [example_dict[str(e_id.item())] for e_id in example_ids]

        def get_sentence_input_info(sentences):
            sentence_tokens_batch = []
            sentence_max_len = 0
            for s in sentences:
                sentence_tokens_batch.append(s.word_tokens())
                s_len = len(s.word_tokens())
                if s_len > sentence_max_len:
                    sentence_max_len = s_len

            sentence_tokens_batch = data_tool.align_mult_sentence_tokens(sentence_tokens_batch, sentence_max_len,
                                                                         self.bert.tokenizer.unk_token, direction='left')
            return sentence_tokens_batch, sentence_max_len

        sentence1s = [e.sentence1 for e in examples]
        sentence2s = [e.sentence2 for e in examples]

        sentence_tokens_batch1, sent_max_len1 = get_sentence_input_info(sentence1s)
        sentence_tokens_batch2, sent_max_len2 = get_sentence_input_info(sentence2s)

        sentence_pair_tokens_batch = torch.tensor([self.bert.tokenizer.encode(s1, s2, add_special_tokens=True)
                                                  for s1, s2 in zip(sentence_tokens_batch1, sentence_tokens_batch2)],
                                                  device=self.device)

        segment_ids = torch.cat([torch.zeros(sent_max_len1 + 2), torch.ones(sent_max_len2 + 1)]).to(
            device=self.device, dtype=torch.long)
        sep_index = sent_max_len1 + 1

        result = {
            'sentence_pair_tokens_batch': sentence_pair_tokens_batch,
            'segment_ids': segment_ids,
            'sep_index': sep_index,
        }
        return result

    def forward(self, *input_data, **kwargs):
        if len(kwargs) == 3: # common run or visualization
            data_batch = kwargs
            sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
            segment_ids = data_batch['segment_ids']
            sep_index = data_batch['sep_index']
        else:
            sentence_pair_tokens, segment_ids, sep_index = input_data

        sentence_pair_reps, sentence1_reps, sentence2_reps = self.bert(sentence_pair_tokens, segment_ids, sep_index)

        # star_time = time.time()
        result = self.semantic_layer(sentence1_reps, sentence2_reps)
        result = torch.cat([sentence_pair_reps, result], dim=1)

        result = self.fully_connection(result)
        return result

    def get_regular_parts(self):
        regular_part_list = ( self.fully_connection, self.bert)
        regular_factor_list = (self.arg_dict['fully_regular'], self.arg_dict['bert_regular'])
        return regular_part_list, regular_factor_list

    def get_input_of_visualize_model(self, example_ids, example_dict):
        data_batch = self.deal_with_example_batch(example_ids[0:1], example_dict)
        sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
        segment_ids = data_batch['segment_ids']
        sep_index = torch.tensor(data_batch['sep_index'], device=self.device, dtype= torch.int)

        input_data = (sentence_pair_tokens, segment_ids, sep_index)

        return input_data

    def count_of_parameter(self):
        with torch.no_grad():
            self.cpu()
            model_list = [self, self.bert, self.fully_connection]
            parameter_counts = []
            weight_counts = []
            bias_counts = []
            parameter_list = []
            weights_list = []
            bias_list = []
            for model_ in model_list:
                parameters_temp = model_.named_parameters()
                weights_list.clear()
                parameter_list.clear()
                bias_list.clear()
                for name, p in parameters_temp:
                    # print(name)
                    parameter_list.append(p.reshape(-1))
                    if name.find('weight') != -1:
                        weights_list.append(p.reshape(-1))
                    if name.find('bias') != -1:
                        bias_list.append(p.reshape(-1))
                parameters = torch.cat(parameter_list, dim=0)
                weights = torch.cat(weights_list, dim=0)
                biases = torch.cat(bias_list, dim=0)
                parameter_counts.append(len(parameters))
                weight_counts.append(len(weights))
                bias_counts.append(len(biases))
            for p_count, w_count, b_count in zip(parameter_counts, weight_counts, bias_counts):
                if p_count != w_count + b_count:
                    raise ValueError

            for kind in (parameter_counts, weight_counts, bias_counts):
                total = kind[0]
                others = kind[1:]
                count_temp = 0
                for other in others:
                    count_temp += other
                if total != count_temp:
                    raise ValueError
            self.to(self.device)

            result = [
                {'name': 'entire', 'total': parameter_counts[0], 'weight': weight_counts[0], 'bias': bias_counts[0]},
                {'name': 'bert', 'total': parameter_counts[1], 'weight': weight_counts[1], 'bias': bias_counts[1]},
                {'name': 'fully', 'total': parameter_counts[2], 'weight': weight_counts[2], 'bias': bias_counts[2]},
            ]

            return result


