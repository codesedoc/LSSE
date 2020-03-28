import torch
from abc import abstractmethod


class Framework(torch.nn.Module):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = self.create_arg_dict()
        self.update_arg_dict(arg_dict)
        self.data_type = self.arg_dict['dtype']
        gpu_id = self.arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)
        self.create_models()

    def update_arg_dict(self, arg_dict):
        for name in arg_dict:
            self.arg_dict[name] = arg_dict[name]

        if self.arg_dict['semantic_compare_func'] == 'wmd':
            self.arg_dict['fully_scales'] = [self.arg_dict['max_sentence_length'] ** 2 + self.arg_dict['bert_hidden_dim'], 2]

        elif self.arg_dict['concatenate_input_for_gcn_hidden']:
            self.arg_dict['fully_scales'][0] += self.arg_dict['gcn_hidden_dim']

        if self.arg_dict['task_type'] == 'regression':
            self.arg_dict['fully_scales'][-1] = 1

    def print_arg_dict(self):
        print("*"*80)
        print("framework args")
        for key, value in self.arg_dict.items():
            print('{}: {}'.format(key, value))
        print("*" * 80)
        print('\n')

    @abstractmethod
    def create_arg_dict(self):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def create_models(self):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def deal_with_example_batch(self, example_ids, example_dict):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def get_regular_parts(self):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def get_input_of_visualize_model(self, example_ids, example_dict):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def count_of_parameter(self):
        raise RuntimeError("have not implemented this abstract method")

    @classmethod
    def framework_name(cls):
        raise RuntimeError("have not implemented this abstract method")

