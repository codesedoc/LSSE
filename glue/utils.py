# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import json
import logging
from torch.optim import Optimizer
import math
import torch
from torch.optim.lr_scheduler import LambdaLR
import utils.file_tool as file_tool
import utils.general_tool as general_tool
import utils.parser_tool as parser_tool


logger = logging.getLogger(__name__)


class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dict):
        tensors = list(tensor_dict.values())
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        result = {}
        for key, tensor in self.tensor_dict.items():
            result[key] = tensor[index]
        return result

    def __len__(self):
        tensors = list(self.tensor_dict.values())
        return tensors[0].size(0)


class InputSentence(object):
    def __init__(self, guid, original):
        self.guid = guid
        self.original = original
        self.parsed_info = None
        self.syntax_info = None

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    #count with ROOT
    def len_of_tokens(self):
        result = len(self.parsed_info['words'].copy())
        # if self.parsed_info['has_root']:
        #     result -= 1
        return result

    def word_tokens(self):
        return self.parsed_info['words'].copy()

    def word_tokens_uncased(self):
        result = self.parsed_info['words'].copy()
        for i, _ in enumerate(result):
            result[i] = result[i].lower()
        return result

    def original_sentence(self):
        return self.original

    def sentence_with_root_head(self):
        return 'root ' + self.original

    def original_sentence_uncased(self):
        return self.original.lower()

    def numeral_dependencies(self):
        return self.syntax_info['dependencies'].copy()


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, id, sent_a, sent_b=None, label=None):
        self.guid = guid
        self.id = id
        self.with_root_head = False
        self.text_a = sent_a.original_sentence()
        # self.text_a = sent_a.sentence_with_root_head()
        if sent_b is not None:

            self.text_b = sent_b.original_sentence()
            # self.text_b = sent_b.sentence_with_root_head()
        else:
            self.text_b = None

        self.sent_a = sent_a
        self.sent_b = sent_b
        self.label = label

    def add_root_to_text(self):
        if not self.with_root_head:
            self.text_a = self.sent_a.sentence_with_root_head()
            self.text_b = self.sent_b.sentence_with_root_head()
            self.with_root_head = True
        else:
            print('already add')

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, args, **kwargs):
        self.args = args
        self.example = kwargs['example']
        self.e_id = self.example.id
        self.input_ids = kwargs['input_ids']
        self.attention_mask = kwargs['attention_mask']
        self.token_type_ids = kwargs['token_type_ids']
        self.label = kwargs['label']

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeaturesWithGCN(InputFeatures):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.adj_matrix_a = parser_tool.dependencies2adj_matrix(self.example.sent_a.syntax_info['dependencies'],
                                                                self.args.dep_kind_count,
                                                                self.args.max_sentence_length)

        self.adj_matrix_b = parser_tool.dependencies2adj_matrix(self.example.sent_b.syntax_info['dependencies'],
                                                                self.args.dep_kind_count,
                                                                self.args.max_sentence_length)
        self.sep_index = kwargs['sep_index']

        self.text_a_len = kwargs['text_a_len']
        self.text_a_org_len = self.example.sent_a.len_of_tokens()
        self.sent_a_id = self.example.sent_a.guid

        self.text_b_len = kwargs['text_b_len']
        self.text_b_org_len = self.example.sent_b.len_of_tokens()

        self.word_piece_flags = kwargs['word_piece_flags']


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    data_path = None

    def __init__(self):
        self.sentence_dict = None
        self.org_sent2sent_obj_dict = None
        self.type2sentence_dict = {'train': None, 'test': None, 'dev': None}
        self.get_all_sentence_dict()

        self.example_dict = None
        self.type2examples = {'train': None, 'test': None, 'dev': None}
        self.get_all_example_dict()

        self._parse_sentences()

    def get_all_sentence_dict(self):
        if self.sentence_dict is None:
            train_sentence_dict = self.get_sentence_dict(set_type='train')
            test_sentence_dict = self.get_sentence_dict(set_type='test')
            dev_sentence_dict = self.get_sentence_dict(set_type='dev')
            sentence_dict_temp = {}
            sentence_dict_temp.update(train_sentence_dict)
            sentence_dict_temp.update(test_sentence_dict)
            sentence_dict_temp.update(dev_sentence_dict)

            self.org_sent2sent_obj_dict = {}
            for sent_id, sent_obj in sentence_dict_temp.items():
                org_sent = sent_obj.original_sentence()
                if org_sent not in self.org_sent2sent_obj_dict:
                    self.org_sent2sent_obj_dict[org_sent] = sent_obj

            self.org_sent2sent_obj_dict = sorted(self.org_sent2sent_obj_dict.items(), key=lambda x: x[1].original_sentence())
            self.org_sent2sent_obj_dict = dict(self.org_sent2sent_obj_dict)

            self.sentence_dict = {}
            count_temp = 0
            for sent_id, sent_obj in self.org_sent2sent_obj_dict.items():
                sent_obj.guid = count_temp
                self.sentence_dict[str(count_temp)] = sent_obj
                count_temp += 1


            if len(self.sentence_dict) != len(self.org_sent2sent_obj_dict):
                raise ValueError

            sent_filename = file_tool.connect_path(self.data_path, 'original_sentence.txt')

            if not file_tool.check_file(sent_filename):
                word_list_filename = file_tool.connect_path(self.data_path, 'sentence_words(bert-base-cased).txt')
                self.output_sentences(sent_filename)
                self.output_words_split_by_tokenizer(word_list_filename)

        return self.sentence_dict

    def get_all_example_dict(self):
        def example_dict_from_examples(examples):
            example_dict = {}
            for e in examples:
                example_dict[str(e.guid)] = e
            if len(examples) != len(example_dict):
                raise ValueError
            return example_dict

        if self.example_dict is None:
            train_examples = self.get_examples(set_type='train')
            test_examples = self.get_examples(set_type='test')
            dev_examples = self.get_examples(set_type='dev')

            self.example_dict = {}
            self.example_dict.update(example_dict_from_examples(train_examples))
            self.example_dict.update(example_dict_from_examples(test_examples))
            self.example_dict.update(example_dict_from_examples(dev_examples))

        return self.example_dict

    def add_root_to_text_of_example(self):
        for e in self.example_dict.values():
            e.add_root_to_text()

    def get_examples(self, data_dir=None, set_type='train'):

        if set_type not in self.type2examples:
            raise ValueError

        if not data_dir:
            data_dir = self.data_path
        data_file = file_tool.connect_path(data_dir, '%s.tsv' % set_type)

        examples = self.type2examples[set_type]
        if examples is None:
            examples = self._create_examples(self._read_tsv(data_file), set_type)
            self.type2examples[set_type] = examples

        return examples

    def get_sentence_dict(self, data_dir=None, set_type='train'):
        if set_type not in self.type2sentence_dict:
            raise ValueError

        if not data_dir:
            data_dir = self.data_path
        data_file = file_tool.connect_path(data_dir, '%s.tsv' % set_type)

        sentence_dict = self.type2sentence_dict[set_type]
        if sentence_dict is None:
            sentence_dict = self._create_sentence_dict(self._read_tsv(data_file), set_type)
            self.type2sentence_dict[set_type] = sentence_dict

        return sentence_dict

    def _parse_sentences(self):
        def check():
            if len(parsed_sentence_dict) != len(self.sentence_dict):
                raise ValueError("parsed_sentence_dict not march sentence_dict")
                pass

            if not general_tool.compare_two_dict_keys(self.sentence_dict.copy(), parsed_sentence_dict.copy()):
                raise ValueError("parsed_sentence_dict not march sentence_dict")

            for sent_id_, info in parsed_sentence_dict.items():
                if info['original'] != self.sentence_dict[sent_id_].original:
                    raise ValueError("parsed_sentence_dict not march sentence_dict")

        parsed_sentence_org_file = file_tool.connect_path(self.data_path, 'parsed_sentences.txt')
        parsed_sentence_dict_file = file_tool.connect_path(self.data_path, 'parsed_sentence_dict.pkl')
        if file_tool.check_file(parsed_sentence_dict_file):
            parsed_sentence_dict = file_tool.load_data_pickle(parsed_sentence_dict_file)
        else:
            parsed_sentence_dict = parser_tool.extra_parsed_sentence_dict_from_org_file(parsed_sentence_org_file)
            file_tool.save_data_pickle(parsed_sentence_dict, parsed_sentence_dict_file)

        check()
        logger.info('Parsed sentences information correct !')

        for sent_id, parsed_info in parsed_sentence_dict.items():
            sent_id = str(sent_id)
            self.sentence_dict[sent_id].parsed_info = parsed_info

        self.parse_info = parser_tool.process_parsing_sentence_dict(parsed_sentence_dict, modify_dep_name=True,
                                                                    delete_root_dep=True)
        numeral_sentence_dict = self.parse_info.numeral_sentence_dict

        if not general_tool.compare_two_dict_keys(self.sentence_dict.copy(), numeral_sentence_dict.copy()):
            raise ValueError("numeral_sentence_dict not march sentence_dict")

        for sent_id in self.sentence_dict.keys():
            self.sentence_dict[sent_id].syntax_info = numeral_sentence_dict[sent_id]

        print('the count of dep type:{}'.format(self.parse_info.dependency_count))
        print('the max len of sentence_tokens:{}'.format(self.parse_info.max_sent_len))

        pass


    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    def _create_sentence_dict(self, lines, set_type):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    def output_sentences(self, filename):
        logger.debug('Output sentences')
        sentence_dict = self.sentence_dict.copy()
        save_data = []
        for sent_id, sent in sentence_dict.items():
            save_data.append('{}\t{}'.format(sent_id, sent.original_sentence()))
        file_tool.save_list_data(save_data, filename, 'w')

    def output_words_split_by_tokenizer(self, filename):
        # mrpc_obj = corpus.mrpc.get_mrpc_obj()
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        general_tool.covert_transformer_tokens_to_words(self, tokenizer, filename, '##')

    def _org_sent2sent_obj_dict(self, sentence_dict):
        result = {}
        for s_id, sent in sentence_dict.items():
            result[sent.original_sentence()] = sent

        if len(result) != len(sentence_dict):
            raise ValueError

        return result

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)