import torch
from abc import abstractmethod
import glue.glue as glue


class Framework(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_args()
        self.create_models()
        self.processor = glue.glue_processors[self.args.task_name]()
        # self.init_weights()

    # def init_weights(self):
    #     self.apply(self._init_weights)

    def update_args(self):
        if self.args.semantic_compare_func == 'wmd':
            self.args.fully_scales = [self.args.max_sentence_length ** 2 + self.args.bert_hidden_dim, 2]

        elif not self.args.without_concatenate_input_for_gcn_hidden:
            self.args.fully_scales[0] += self.args.gcn_hidden_dim

        if self.args.output_mode == 'regression':
            self.args.fully_scales[-1] = 1

    def _count_of_parameter(self, *, model_list, name_list):
        if len(model_list) != len(name_list):
            raise ValueError

        with torch.no_grad():
            self.cpu()
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
            self.to(self.args.device)
            result = []
            for n, t, w, b in zip(name_list, parameter_counts, weight_counts, bias_counts):
                result.append({'name': n, 'total': t, 'weight': w, 'bias': b})

            return tuple(result)

    @abstractmethod
    def _init_weights(self, module):
        raise RuntimeError("have not implemented this abstract method")
        # if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
        # elif isinstance(module, torch.nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, torch.nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

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

