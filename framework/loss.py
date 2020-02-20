import torch


class BiCELo(torch.nn.Module):
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
