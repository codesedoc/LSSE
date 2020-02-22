import torch
from abc import abstractmethod


class Loss(torch.nn.Module):
    def __init__(self, arg_dict, regular_part_list, regular_factor_list):
        super().__init__()
        self.arg_dict = arg_dict
        self.regular_flag = arg_dict['regular_flag']
        self.regular_part_list = regular_part_list
        self.regular_factor_list = regular_factor_list

    def forward(self, model_outputs, labels):
        model_outputs = model_outputs.reshape(labels.size())
        cross_loss = torch.nn.BCELoss()(model_outputs, labels)
        batch_size = self.arg_dict['batch_size']
        regular_items = []
        if self.regular_flag:
            weights_list = []
            for part, factor in zip(self.regular_part_list, self.regular_factor_list):
                parameters_temp = part.named_parameters()
                weights_list.clear()
                for name, p in parameters_temp:
                    # print(name)
                    if (name.startswith('w') or name.find('weight') != -1) and (name.find('bias') == -1):
                        weights_list.append(p.reshape(-1))
                weights = torch.cat(weights_list, dim=0)
                # print(len(weights))
                para_sum = torch.pow(weights, 2).sum()
                regular_items.append((factor*para_sum)/(2*batch_size))
        result = cross_loss
        for item in regular_items:
            result += item
        correct_count = 0
        for output_, label in zip(model_outputs, labels):
            if torch.abs(output_-label) < 0.5:
                correct_count += 1
        return result, correct_count


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

