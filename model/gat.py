import torch.nn as nn
import torch
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, args, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.args = args
        self.dropout = self.args.gat_dropout
        self.in_features = self.args.gcn_hidden_dim
        self.out_features = self.args.gcn_hidden_dim
        self.alpha = self.args.gat_alpha
        # self.concat = concat

        # self.weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.ones((self.in_features, self.out_features), device=self.args.device)), requires_grad=True)
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attention_weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.ones((2*self.out_features, 1), device=self.args.device)), requires_grad=True)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.zero_vec = -9e31 * torch.ones([self.args.train_batch_size, self.args.max_sentence_length, self.args.max_sentence_length], device=self.args.device)

    def forward(self, input_, adj, weights):
        h = torch.bmm(input_, weights)
        N = h.size()[1]
        cur_batch_size = h.size()[0]
        a_w1 = self.attention_weight[:self.out_features, :]
        a_w2 = self.attention_weight[self.out_features:, :]
        # a_input = torch.cat([h.repeat(1, 1, N).view(cur_batch_size, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(cur_batch_size, N, -1, 2 * self.out_features)
        # e_old = self.leakyrelu(torch.matmul(a_input, self.attention_weight).squeeze(-1))
        a1 = torch.matmul(h, a_w1)
        a2 = torch.transpose(torch.matmul(h, a_w2), -1, -2)

        e = self.leakyrelu(a1.expand(-1, N, N) + a2.expand(-1, N, N))

        attention = torch.where(adj > 0, e, self.zero_vec[:cur_batch_size])
        attention = F.softmax(attention, dim=-2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime

        return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
