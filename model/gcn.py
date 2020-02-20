import torch


class GCN(torch.nn.Module):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.layer = arg_dict['gcn_layer']
        self.dep_kind_count = arg_dict['dep_kind_count']
        # self.relation_number = arg_dict['gcn_relation_number']
        self.gate_flag = arg_dict['gcn_gate_flag']
        self.norm_item = arg_dict['gcn_norm_item']
        self.self_loop_flag = arg_dict['gcn_self_loop_flag']
        # word_embedding_dim = arg_dict['word_embedding_dim']
        self.hidden_dim = arg_dict['gcn_hidden_dim']
        self.group_layer_limit_flag = arg_dict['group_layer_limit_flag']
        if self.group_layer_limit_flag:
            self.dep_layer_limit_list = arg_dict['dep_layer_limit_list']
        gpu_id = arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

        self.weight_in_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                        (self.hidden_dim, self.hidden_dim)))

        self.weight_out_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                         (self.hidden_dim, self.hidden_dim)))

        self.bias_in_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                      (self.hidden_dim, 1)))

        self.bias_out_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                       (self.hidden_dim, 1)))

        if self.gate_flag:
            self.weight_gate_in_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                                 (self.hidden_dim, self.hidden_dim)))
            self.weight_gate_out_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                                  (self.hidden_dim, self.hidden_dim)))

            self.bias_gate_in_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                               (self.hidden_dim, 1)))
            self.bias_gate_out_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                                (self.hidden_dim, 1)))

        if self.self_loop_flag:
            self.weight_loop_list = torch.nn.ParameterList(self.get_para_list(self.layer,
                                                                              (self.hidden_dim, self.hidden_dim)))

    def get_para_list(self, length, shape):
        return [
            torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.ones(*shape, device=self.device, requires_grad=True)))
            for i in range(length)]

    def forward(self, sentences, adj_matrix):
        def calculate_convolution(weight_gate_list, bias_gate_list, weight_list, bias_list, index, d_kind, adj_matrix,
                                  h):
            if self.gate_flag:
                z = torch.baddbmm(bias_gate_list[index].expand([current_batch_size, -1, -1]),
                                  batch1=weight_gate_list[index].expand([current_batch_size, -1, -1]),
                                  batch2=h)
                gate = torch.sigmoid(z)
            else:
                gate = 1

            relation = self.norm_item * gate * torch.baddbmm(
                bias_list[index].expand([current_batch_size, -1, -1]),
                batch1=weight_list[index].expand([current_batch_size, -1, -1]),
                batch2=h)
            relation = torch.bmm(relation, adj_matrix[:, d_kind].type(relation.dtype))
            return relation

        adj_matrix_in = adj_matrix
        adj_matrix_out = adj_matrix.permute(0, 1, 3, 2)
        hidden = sentences.permute(0, 2, 1)
        current_batch_size = hidden.size()[0]
        for layer in range(self.layer):
            # print()
            # print(layer)
            if self.self_loop_flag:
                sum_ = torch.bmm(self.weight_loop_list[layer].expand([current_batch_size, -1, -1]), hidden)
            else:
                sum_ = 0
            for dep_kind in range(self.dep_kind_count):
                if self.group_layer_limit_flag:
                    if layer + 1 > self.dep_layer_limit_list[dep_kind]:
                        # print([layer, dep_kind, self.dep_layer_limit_list[dep_kind]])
                        continue
                para_list_index = layer * self.dep_kind_count + dep_kind

                relation_in = calculate_convolution(self.weight_gate_in_list, self.bias_gate_in_list,
                                                    self.weight_in_list, self.bias_in_list, para_list_index,
                                                    dep_kind, adj_matrix_in, hidden)

                relation_out = calculate_convolution(self.weight_gate_out_list, self.bias_gate_out_list,
                                                     self.weight_out_list, self.bias_out_list, para_list_index,
                                                     dep_kind, adj_matrix_out, hidden)

                result2 = relation_in + relation_out
                # if (result2 != result1).any():
                #     raise ValueError

                sum_ += result2

            hidden = torch.relu(sum_)

        return hidden.permute(0, 2, 1)


class GCNUndir(torch.nn.Module):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.layer = arg_dict['gcn_layer']
        self.dep_kind_count = arg_dict['dep_kind_count']
        # self.relation_number = arg_dict['gcn_relation_number']
        self.gate_flag = arg_dict['gcn_gate_flag']
        self.norm_item = arg_dict['gcn_norm_item']
        self.self_loop_flag = arg_dict['gcn_self_loop_flag']
        # word_embedding_dim = arg_dict['word_embedding_dim']
        self.hidden_dim = arg_dict['gcn_hidden_dim']
        self.group_layer_limit_flag = arg_dict['group_layer_limit_flag']
        if self.group_layer_limit_flag:
            self.dep_layer_limit_list = arg_dict['dep_layer_limit_list']
        gpu_id = arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

        self.weight_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                     (self.hidden_dim, self.hidden_dim)))
        self.bias_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                   (self.hidden_dim, 1)))

        if self.gate_flag:
            self.weight_gate_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                              (self.hidden_dim, self.hidden_dim)))
            self.bias_gate_list = torch.nn.ParameterList(self.get_para_list(self.layer * self.dep_kind_count,
                                                                            (self.hidden_dim, 1)))

        if self.self_loop_flag:
            self.weight_loop_list = torch.nn.ParameterList(self.get_para_list(self.layer,
                                                                              (self.hidden_dim, self.hidden_dim)))

    def get_para_list(self, length, shape):
        return [
            torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.ones(*shape, device=self.device)), requires_grad=True)
            for i in range(length)]

    def forward(self, sentences, adj_matrix):
        adj_matrix_in = adj_matrix
        adj_matrix_out = adj_matrix.permute(0, 1, 3, 2)
        hidden = sentences.permute(0, 2, 1).type(torch.float32)
        current_batch_size = hidden.size()[0]
        for layer in range(self.layer):
            # print()
            # print(layer)
            if self.self_loop_flag:
                sum_ = torch.bmm(self.weight_loop_list[layer].expand([current_batch_size, -1, -1]), hidden)
            else:
                sum_ = 0
            for dep_kind in range(self.dep_kind_count):
                if self.group_layer_limit_flag:
                    if layer + 1 > self.dep_layer_limit_list[dep_kind]:
                        # print([layer, dep_kind, self.dep_layer_limit_list[dep_kind]])
                        continue
                if self.gate_flag:
                    para_list_index = layer * self.dep_kind_count + dep_kind
                    z = torch.baddbmm(self.bias_gate_list[para_list_index].expand([current_batch_size, -1, -1]),
                                      batch1=self.weight_gate_list[para_list_index].expand(
                                          [current_batch_size, -1, -1]), batch2=hidden)
                    gate = torch.sigmoid(z)
                    relation_in = self.norm_item * gate * torch.baddbmm(
                        self.bias_list[para_list_index].expand([current_batch_size, -1, -1]),
                        batch1=self.weight_list[para_list_index].expand([current_batch_size, -1, -1]),
                        batch2=hidden)
                    relation_in = torch.bmm(relation_in, adj_matrix_in[:, dep_kind].type(relation_in.dtype))
                    relation_out = self.norm_item * gate * torch.baddbmm(
                        self.bias_list[para_list_index].expand([current_batch_size, -1, -1]),
                        batch1=self.weight_list[para_list_index].expand([current_batch_size, -1, -1]),
                        batch2=hidden)
                    relation_out = torch.bmm(relation_out, adj_matrix_out[:, dep_kind].type(relation_out.dtype))
                    sum_ += relation_in + relation_out
            hidden = torch.relu(sum_)

        return hidden.permute(0, 2, 1)