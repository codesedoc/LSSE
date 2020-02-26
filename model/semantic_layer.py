import torch
import math


class SemanticLayer(torch.nn.Module):
    def __init__(self, arg_dict):
        self.arg_dict = arg_dict
        super().__init__()

    def revise_zero_data(self, tensor):
        boundary = 0.00001 ** 2
        flag_tensor = (tensor < boundary).type(torch.float)
        tensor = tensor + flag_tensor * boundary
        return tensor

    def forward(self, sentence1s, sentence2s):
        sentence1s = sentence1s.float()
        sentence2s = sentence2s.float()
        if self.arg_dict['semantic_compare_func'] == 'l2':
            result = self.l2distance(sentence1s, sentence2s)

        elif self.arg_dict['semantic_compare_func'] == 'arccos':
            result = self.calculate_arc_cos(sentence1s, sentence2s)
        else:
            raise ValueError

        if torch.isnan(result).sum() > 0:
            print(torch.isnan(result))
            raise ValueError
        return result

    def l2distance(self, sentence1s, sentence2s):
        sentence1_len = sentence1s.size()[1]
        sentence2_len = sentence2s.size()[1]
        sentence1s = sentence1s.sum(dim=1) / sentence1_len
        sentence2s = sentence2s.sum(dim=1) / sentence2_len

        result = torch.pow(sentence1s - sentence2s, 2)
        if (result == 0).any():
            result = self.revise_zero_data(result)
        result = torch.sqrt(result)

        # result = torch.abs(sentence1s - sentence2s)

        return result

    def calculate_arc_cos(self, input_data1, input_data2):
        shape1 = input_data1.size()
        shape2 = input_data2.size()
        result = []
        for i in range(shape1[0]):
            x = input_data1[i]
            y = input_data2[i]
            result_temp = torch.nn.functional.normalize(x, dim=1).mm(
                torch.nn.functional.normalize(y, dim=1).T).squeeze()
            if torch.isnan(result_temp).sum() > 0:
                print(torch.isnan(result_temp))
                raise ValueError
            # l_length, c_length = result_temp.size()
            index_list = (result_temp >=1 ).nonzero().cpu().numpy()
            index_list = [np.array(index_pair).reshape(-1,1) for index_pair in index_list]
            if len(index_list)>0:
                index_list = np.concatenate(index_list, axis=1)
                result_temp[index_list] -= 1e-6

            index_list = (result_temp <= -1).nonzero().cpu().numpy()
            index_list = [np.array(index_pair).reshape(-1, 1) for index_pair in index_list]
            if len(index_list) > 0:
                index_list = np.concatenate(index_list, axis=1)
                result_temp[index_list] += 1e-6

            # for l in range(l_length):
            #     for c in range(c_length):
            #         if result_temp[l, c] >= 1:
            #             result_temp[l, c] = 1.0 - 1e-8
            #         if result_temp[l, c] <= -1:
            #             result_temp[l, c] = -1.0 + 1e-8
            if (torch.abs(result_temp) >=1 ).any():
                raise ValueError

            result_temp = torch.acos(result_temp) / math.pi
            result_temp = torch.ones_like(result_temp) - result_temp
            result_temp = result_temp.reshape(-1)
            result.append(result_temp)
        result = torch.stack(result, dim=0)

        return result