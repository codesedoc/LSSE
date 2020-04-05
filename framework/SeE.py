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

    def __init__(self, args):
        args.fully_scales = [args.gcn_hidden_dim * 2, 2]
        super().__init__(args)
        self.name = SeE.name
        self.result_path = SeE.result_path


    @classmethod
    def framework_name(cls):
        return cls.name

    def create_models(self):
        self.with_linear_head = False
        if self.with_linear_head:
            self.encoder = BertForSeqClassify(self.args)
        else:
            self.encoder = BertBase(self.args)
            config = self.encoder.config
            self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
            self.classifier.load_state_dict(torch.load('model/classifier.pts'))
            pass
            # self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
            # self.classifier.bias.data.zero_()

    def update_args(self):
        super().update_args()
        if self.args.do_train:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            output_dir = file_tool.connect_path(self.result_path,
                                                'train',
                                                'bs:{}-lr:{}'.format(self.args.per_gpu_train_batch_size,
                                                                     self.args.learning_rate),
                                                time_str)

            file_tool.makedir(output_dir)
            if not file_tool.check_dir(output_dir):
                raise RuntimeError
            self.args.output_dir = output_dir

    def count_of_parameter(self):
        if not self.with_linear_head:
            model_list = [self, self.encoder, self.classifier]
            name_list = ['Entire', self.encoder.name, 'classifier']
        else:
            model_list = [self, self.encoder]
            name_list = [self, self.encoder.name]

        result = self._count_of_parameter(model_list=model_list, name_list=name_list)
        return result

    def forward(self, **kwargs):
        if len(kwargs) == 0:
            raise ValueError

        input_ids_batch = kwargs['input_ids']
        token_type_ids_batch = kwargs['token_type_ids']
        attention_mask_batch = kwargs['attention_mask']
        labels = kwargs['labels']

        if not self.with_linear_head:
            last_hidden_states, pooled_output = self.encoder(input_ids_batch, token_type_ids_batch, attention_mask_batch)
            output = self.dropout(pooled_output)
            predicts = self.classifier(output)
            if self.args.num_labels > 1:
                if torch.isnan(last_hidden_states).sum() > 0:
                    print(torch.isnan(last_hidden_states))
                    raise ValueError
                loss = torch.nn.CrossEntropyLoss()(predicts.view(-1, 2), labels.view(-1))
            elif self.args.num_labels == 1:
                loss = torch.nn.MSELoss()(predicts.view(-1), labels.view(-1))
            else:
                raise ValueError

            outputs = loss, predicts

        else:
            outputs = self.encoder(input_ids_batch, token_type_ids_batch, attention_mask_batch, labels)
        return outputs


