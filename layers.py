import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        print('>>> conv_input: ', conv_input, '\n')
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat  # 用于标记是否是最后一层
        self.nrela_dim = nrela_dim
        ########## liyirui replace this ##########
        # <begin>
        # self.a = nn.Parameter(torch.zeros(
        #     size=(out_features, 2 * in_features + nrela_dim)))
        # <end>
        # <new>
        self.a = nn.Parameter(torch.zeros(
             size=(out_features, 2 * in_features)))
        # <end new>
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]  # WN18RR: 40943

        # print(">>>")
        # print(edge.size())              # WN18RR: torch.Size([2, 86835])
        # print(edge_list_nhop.size())    # WN18RR: torch.Size([2, 207376])
        # print(edge)
        # print("<<<")
        
        # 这里是论文中公式(5)(6)(7)所对应的代码
        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)  # WN18RR: torch.Size([294211, 50])
        
        ########## liyirui's improve is here ##########
        # <begin>
        # add self loop into edge
        edge_s_loop_0 = torch.linspace(start=0, end=input.size()[0]-1, steps=input.size()[0]).long().cuda()
        edge_s_loop_0 = torch.unsqueeze(edge_s_loop_0, 1)
        edge_s_loop = torch.cat((edge_s_loop_0[:, :], edge_s_loop_0[:, :]), dim=1).transpose(0, 1)  # WN18RR: torch.Size([2, 40943])
        
        edge = torch.cat((edge[:, :], edge_s_loop[:, :]), dim=1)  # WN18RR: torch.Size([2, 335154])
        
        # add self loop embed into edge_embed
        self_loop = torch.zeros(input.size()).cuda()  # WN18RR: torch.Size([40943, 50])
        edge_embed = torch.cat((edge_embed[:, :], self_loop[:, :]), dim=0)  # WN18RR: torch.Size([335154, 50])

        # <end>
        # print(">>>")
        # print(input[edge[0, :], :]) 
        # print(input[edge[1, :], :])
        # print(edge_embed[:, :])
        # print("<<<")

        # print(">>>")
        # print(edge.size())                  # WN18RR: torch.Size([2, 294211]) one_hop 86835 + nhop 207376
        # print(input.size())                 # WN18RR: torch.Size([40943, 50])
        # print(input[edge[0, :], :].size())  # WN18RR: torch.Size([294211, 50])
        # print(input[edge[1, :], :].size())  # WN18RR: torch.Size([294211, 50])
        # print(edge_embed[:, :].size())      # WN18RR: torch.Size([294211, 50])
        # print("<<<")

        # 公式(5) 
        # W_1: a
        # h_i: input[edge[0, :], :]
        # h_j: input[edge[1, :], :]
        # g_k: edge_embed[:, :]
        ########## liyirui comment this line ##########
        # <begin>
        # edge_h = torch.cat(
        #     (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()  # WN18RR: torch.Size([100, 294211])
        # <end>
        # edge_h: (2*in_dim + nrela_dim) x E

        ########## liyirui's improve is here ##########
        # <begin>
        edge_h = torch.cat((input[edge[1, :], :], edge_embed[:, :]), dim=1).transpose(0, 1) # WN18RR: torch.Size([100, 335154])
        print(edge_h.size())
        # <end>

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        # 公式(6)
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        # 公式(7) begin
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D
        # 公式(8)
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
