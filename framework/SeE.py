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
            'sentence_max_len_for_bert': 128,
            'pad_on_right': True,
            'dtype': torch.float32,
        }
        return arg_dict

    def create_models(self):
        self.with_linear_head = False
        if self.with_linear_head:
            self.bert = ALBertForSeqClassify()
        else:
            self.bert = BertBase()
            config = self.bert.config
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            self.classifier = torch.nn.Linear(config.hidden_size, 2)
            self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.classifier.bias.data.zero_()
        self.bert_name = self.bert.name

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

    def count_of_parameter(self):
        with torch.no_grad():
            self.cpu()
            if not self.with_linear_head:
                model_list = [self, self.bert, self.classifier]
            else:
                model_list = [self, self.bert]

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
                {'name': self.bert_name, 'total': parameter_counts[1], 'weight': weight_counts[1], 'bias': bias_counts[1]},
            ]
            if not self.with_linear_head:
                result.append({'name': 'classifier', 'total': parameter_counts[2], 'weight': weight_counts[2],'bias': bias_counts[2]})

            return result

    def deal_with_example_batch(self, example_ids, example_dict):
        examples = [example_dict[str(e_id.item())] for e_id in example_ids]
        sentence_max_len = self.arg_dict['sentence_max_len_for_bert']
        pad_on_right = self.arg_dict['pad_on_right']
        pad_token = self.bert.tokenizer.convert_tokens_to_ids([self.bert.tokenizer.pad_token])[0]
        mask_padding_with_zero = True
        pad_token_segment_id = 0

        sentence1s = [e.sentence1 for e in examples]
        sentence2s = [e.sentence2 for e in examples]
        labels = torch.tensor([e.label for e in examples], dtype=torch.long, device=self.device)

        input_ids_batch = []
        token_type_ids_batch = []
        attention_mask_batch = []

        for s1, s2 in zip(sentence1s, sentence2s):
            inputs_ls_cased = self.bert.tokenizer.encode_plus(s1.word_tokens(), s2.word_tokens(),
                                                                  add_special_tokens=True,
                                                                  max_length=sentence_max_len, )
            input_ids, token_type_ids = inputs_ls_cased["input_ids"], inputs_ls_cased["token_type_ids"]

            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = sentence_max_len - len(input_ids)
            if not pad_on_right:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            input_ids_batch.append(input_ids)
            token_type_ids_batch.append(token_type_ids)
            attention_mask_batch.append(attention_mask)

        input_ids_batch = torch.tensor(input_ids_batch, device=self.device)
        token_type_ids_batch = torch.tensor(token_type_ids_batch, device=self.device)
        attention_mask_batch = torch.tensor(attention_mask_batch, device=self.device)
        result = {
            'input_ids_batch': input_ids_batch,
            'token_type_ids_batch': token_type_ids_batch,
            'attention_mask_batch': attention_mask_batch,
            'labels': labels
        }
        return result

    def forward(self, *input_data, **kwargs):
        if len(kwargs) > 0: # common run or visualization
            data_batch = kwargs
            input_ids_batch = data_batch['input_ids_batch']
            token_type_ids_batch = data_batch['token_type_ids_batch']
            attention_mask_batch = data_batch['attention_mask_batch']
            labels = data_batch['labels']
        else:
            input_ids_batch, token_type_ids_batch, attention_mask_batch, labels = input_data

        if not self.with_linear_head:
            last_hidden_states, pooled_output = self.bert(input_ids_batch, token_type_ids_batch, attention_mask_batch)
            output = self.dropout(pooled_output)
            predicts = self.classifier(output)
            if torch.isnan(last_hidden_states).sum() > 0:
                print(torch.isnan(last_hidden_states))
                raise ValueError

            loss = torch.nn.CrossEntropyLoss()(predicts.view(-1, 2), labels.view(-1))

            predicts = np.array(predicts.detach().cpu().numpy()).argmax(axis=1)
        else:
            loss, outputs = self.bert(input_ids_batch, token_type_ids_batch, attention_mask_batch, labels)
            predicts = outputs
            loss = torch.nn.CrossEntropyLoss()(predicts.view(-1, 2), labels.view(-1))
            predicts = np.array(predicts.detach().cpu().numpy()).argmax(axis=1)
        return loss, predicts


