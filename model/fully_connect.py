import torch


class FullyConnection(torch.nn.Module):
    def __init__(self, arg_dict):
        super().__init__()
        layer_list = []
        active_list = []
        scales = arg_dict['fully_scales']
        for i in range(1, len(scales)):
            layer_list.append(torch.nn.Linear(scales[i - 1], scales[i], bias=True))
            active_list.append(torch.nn.Tanh())
        active_list[-1] = torch.nn.Sigmoid()
        self.layer_list = torch.nn.ModuleList(layer_list)
        self.active_list = torch.nn.ModuleList(active_list)
        gpu_id = arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

    def forward(self, input_data):
        # input_data = input_data.permute(1, 0, 2)
        result = input_data
        if torch.isnan(result).sum() > 0:
            print(torch.isnan(result))
            raise ValueError
        for model_, active in zip(self.layer_list, self.active_list):
            result = model_(result)
            result = active(result)

        return result