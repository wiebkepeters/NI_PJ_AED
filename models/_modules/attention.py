from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['ATTENTION']

class ATTENTION(nn.Module):
    def __init__(self,
        dim_cnn: int,
        dim_rnn: int,
        # num_frames: int,
        nb_classes: int
        # h_fn: Callable

    ) -> None:

        super().__init__()

        dim = dim_cnn + dim_rnn
        # self.T = num_frames
        # self.h_fn : Callable = h_fn

        self.W1 : nn.Module = nn.Linear(dim_cnn, dim_cnn, bias=False)
        self.W2 : nn.Module = nn.Linear(dim_rnn, dim_rnn, bias=False)
        self.Wgat1 : nn.Module = nn.Linear(dim, dim)
        self.Wgat1.weight = nn.parameter.Parameter(torch.eye(dim, dim), requires_grad=False)
        self.U : nn.Module = nn.Linear(dim, 1, bias=False)

        self.W3 : nn.Module = nn.Linear(dim_cnn, dim_cnn)
        self.V : nn.Module = nn.Linear(dim_cnn, 1, bias=False)
        self.Wglobal : nn.Module = nn.Linear(dim_rnn, dim_rnn, bias=False)



    def forward(self, feats: torch.Tensor, h_head: torch.Tensor):


        W1_yT = self.W1(feats) # B x T x cnn
        W2_hT = self.W2(h_head) # B x T x rnn

        c_T = torch.cat((W1_yT, W2_hT), dim=-1) # B x T x (cnn+rnn)
        alpha_gat_T = torch.sigmoid(self.U(torch.tanh(self.Wgat1(c_T)))) # B x T x 1

        z_T = alpha_gat_T * feats # B x T x dim_cnn
        beta_lat_T = torch.sigmoid(self.V(self.W3(z_T))) # B x T x 1
        f_T = beta_lat_T * z_T # B x T x dim_cnn
        f_bar = f_T.mean(dim=1)# B x dim_cnn # (-1, feats.shape[1], -1) # B x T x dim_cnn

        # a_T = torch.cat((f_bar, self.Wglobal(h_head)), dim=-1) # B x T x (dim_cnn+dim_rnn)
        a_T = torch.cat((f_bar, self.Wglobal(h_head[:, 0, :])), dim=-1) # B x (dim_cnn+dim_rnn)

        # return f_bar, alpha_gat_T, beta_lat_T

        return a_T, alpha_gat_T, beta_lat_T



