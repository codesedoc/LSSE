import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *


class LSSEFramework(fr.Framework):
    def __init__(self, arg_dict):
        super().__init__(arg_dict)

    def create_arg_dict(self):
        arg_dict = {
            # 'sgd_momentum': 0.4,
            'semantic_compare_func': 'l2',
            'concatenate_input_for_gcn_hidden': True,
            'fully_scales': [768 * 2, 150, 1],
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
            'model_path': "result/LSSE",
        }
        return arg_dict

    def update_arg_dict(self, arg_dict):
        super().update_arg_dict(arg_dict)
        if self.arg_dict['concatenate_input_for_gcn_hidden']:
            self.arg_dict['fully_scales'][0] += self.arg_dict['gcn_hidden_dim']

        if self.arg_dict['repeat_train']:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if self.arg_dict['group_layer_limit_flag']:
                gl = self.arg_dict['group_layer_limit_list']
            else:
                gl = self.arg_dict['gcn_layer']

            model_dir = file_tool.connect_path(self.arg_dict['model_path'], 'bs:{}-lr:{}-gl:{}-gr:{}-fr:{}-br:{}-com_fun:{}'.
                                               format(self.arg_dict['batch_size'], self.arg_dict['learn_rate'], gl,
                                                      self.arg_dict['gcn_regular'],
                                                      self.arg_dict['fully_regular'], self.arg_dict['bert_regular'],
                                                      self.arg_dict['semantic_compare_func']), time_str)

            file_tool.makedir(model_dir)
            if not file_tool.check_dir(model_dir):
                raise RuntimeError
            self.arg_dict['model_path'] = model_dir

    def create_models(self):
        self.bert = BertBase()
        self.gcn = GCNUndir(self.arg_dict)
        self.semantic_layer = SemanticLayer(self.arg_dict)
        self.fully_connection = FullyConnection(self.arg_dict)

    def deal_with_example_batch(self, examples):
        def get_sentence_input_info(sentences):
            sentence_tokens_batch = []
            adj_matrixs = []
            sentence_max_len = 0
            for s in sentences:
                sentence_tokens_batch.append(s.word_tokens())
                adj_matrixs.append(parser_tool.dependencies2adj_matrix(s.syntax_info['dependencies'],
                                                                       self.arg_dict['dep_kind_count'],
                                                                       self.arg_dict['max_sentence_length']))
                s_len = len(s.word_tokens())
                if s_len > sentence_max_len:
                    sentence_max_len = s_len

            sentence_tokens_batch = data_tool.align_mult_sentence_tokens(sentence_tokens_batch, sentence_max_len,
                                                                         self.bert.tokenizer.unk_token)
            adj_matrixs = np.array(adj_matrixs)
            adj_matrixs = adj_matrixs[..., 0:sentence_max_len, 0:sentence_max_len]

            return sentence_tokens_batch, adj_matrixs, sentence_max_len

        sentence1s = [e.sentence1 for e in examples]
        sentence2s = [e.sentence2 for e in examples]

        sentence_tokens_batch1, adj_matrix1s, sent_max_len1 = get_sentence_input_info(sentence1s)
        sentence_tokens_batch2, adj_matrix2s, sent_max_len2 = get_sentence_input_info(sentence2s)

        sentence_pair_tokens_batch = torch.tensor([self.bert.tokenizer.encode(s1, s2, add_special_tokens=True)
                                                  for s1, s2 in zip(sentence_tokens_batch1, sentence_tokens_batch2)],
                                                  device=self.device)

        segment_ids = torch.cat([torch.zeros(sent_max_len1 + 2), torch.ones(sent_max_len2 + 1)]).to(
            device=self.device, dtype=torch.long)
        sep_index = sent_max_len1 + 1

        adj_matrix1s = torch.from_numpy(adj_matrix1s).to(device=self.device, dtype=self.data_type)
        adj_matrix2s = torch.from_numpy(adj_matrix2s).to(device=self.device, dtype=self.data_type)
        result = {
            'adj_matrix1s': adj_matrix1s,
            'adj_matrix2s': adj_matrix2s,
            'sentence_pair_tokens_batch': sentence_pair_tokens_batch,
            'segment_ids': segment_ids,
            'sep_index': sep_index,
        }
        return result

    def forward(self, example_ids, example_dict):
        examples = [example_dict[str(e_id.item())] for e_id in example_ids]

        data_batch = self.deal_with_example_batch(examples)
        sentence_pair_tokens = data_batch['sentence_pair_tokens_batch']
        segment_ids = data_batch['segment_ids']
        sep_index = data_batch['sep_index']
        adj_matrix1s = data_batch['adj_matrix1s']
        adj_matrix2s = data_batch['adj_matrix2s']
        sentence_pair_reps, sentence1_reps, sentence2_reps = self.bert(sentence_pair_tokens, segment_ids, sep_index)

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
        result = torch.cat([sentence_pair_reps, result], dim=1)

        result = self.fully_connection(result)
        return result

    # def run_batch(self, batch):
    #     raise RuntimeError("have not implemented this abstract method")

    def get_regular_parts(self):
        regular_part_list = (self.gcn, self.fully_connection, self.bert)
        regular_factor_list = (self.arg_dict['gcn_regular'], self.arg_dict['fully_regular'], self.arg_dict['bert_regular'])
        return regular_part_list, regular_factor_list

    def get_input_of_visualize_model(self, example_ids, example_dict):
        examples = [example_dict[str(e_id)] for e_id in example_ids]

        data_batch = self.deal_with_example_batch(examples)
        sentence_pair_tokens = data_batch['sentence_pair_tokens']
        segment_ids = data_batch['segment_ids']
        sep_index = data_batch['sep_index']
        adj_matrix1s = data_batch['adj_matrix1s']
        adj_matrix2s = data_batch['adj_matrix2s']

        return sentence_pair_tokens, segment_ids, sep_index, adj_matrix1s, adj_matrix2s

    def count_of_parameter(self):
        with torch.no_grad():
            self.cpu()
            model_list = [self, self.bert, self.gcn, self.fully_connection]
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
                {'name': 'gcn', 'total': parameter_counts[2], 'weight': weight_counts[2], 'bias': bias_counts[2]},
                {'name': 'fully', 'total': parameter_counts[3], 'weight': weight_counts[3], 'bias': bias_counts[3]},
            ]

            return result


