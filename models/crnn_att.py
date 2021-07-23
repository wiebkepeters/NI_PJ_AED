# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module, GRUCell, Linear
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from ._modules import DNN
from ._modules import attention

__all__ = ['CRNN_ATT']


class CRNN_ATT(Module):

    def __init__(self,
                 cnn_channels: int,
                 cnn_dropout: float,
                 rnn_in_dim: int,
                 rnn_out_dim: int,
                 nb_classes: int) \
            -> None:
        """CRNN model.

        :param cnn_channels: Amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param rnn_in_dim: Input dimensionality of the RNN.
        :type rnn_in_dim: int
        :param rnn_out_dim: Output dimensionality of the RNN.
        :type rnn_out_dim: int
        :param nb_classes: Amount of classes to be predicted.
        :type nb_classes: int
        """
        super().__init__()

        self.rnn_in_dim: int = rnn_in_dim
        self.rnn_hh_size: int = rnn_out_dim*2
        self.nb_classes: int = nb_classes

        self.dnn: Module = DNN(
            cnn_channels=cnn_channels,
            cnn_dropout=cnn_dropout)

        # self.rnn: Module = GRUCell(
        #     input_size=rnn_in_dim,
        #     hidden_size=self.rnn_hh_size,
        #     bias=True)

        self.rnn: Module = torch.nn.GRU(
            input_size=rnn_in_dim,
            hidden_size=(self.rnn_hh_size//2),
            batch_first=True,
            bidirectional=True
        )

        self.attention: Module = attention.ATTENTION(
                    dim_cnn=self.rnn_in_dim,
                    dim_rnn=self.rnn_hh_size, # *2: bi-directional GRU cell
                    nb_classes=self.nb_classes
                )

        self.classifier: Module = nn.Linear(rnn_in_dim+self.rnn_hh_size, nb_classes)
        # self.classifier2 : Module = nn.Linear(1, nb_classes)

    def forward(self,
                    x: Tensor) \
                -> Tensor:
            """Forward pass of the CRNN model.

            :param x: Input to the CRNN.
            :type x: torch.Tensor
            :return: Output predictions.
            :rtype: torch.Tensor
            """
            b_size, t_steps, _ = x.size()

            feats: Tensor = self.dnn(x).permute(
                0, 2, 1, 3
            ).reshape(b_size, t_steps, -1)

            # h: Tensor = torch.zeros(
            #     b_size, self.rnn_hh_size
            # ).to(x.device)

            # h_head: Tensor = torch.zeros(
            #     b_size, self.rnn_hh_size
            # ).to(x.device)

            # for t_step in feats.permute(1, 0, 2):
            #     h = self.rnn(t_step, h)
            #     h_head += h

            # h_head_mean = h_head / t_steps # mean rnn output
            # h_head_mean = h_head_mean.unsqueeze(1).expand(-1, t_steps,-1)

            h_head, _ = self.rnn(feats) # B x T x (2*rnn_out)

            # f_bar, alpha, beta = self.attention(feats, h_head_mean)
            # outputs = self.classifier2(beta)

            # a_T, alpha, beta = self.attention(feats, h_head_mean)
            a_T, alpha, beta = self.attention(feats, h_head.mean(dim=1, keepdim=True).expand(-1, t_steps, -1))
            outputs = self.classifier(a_T)

            return outputs, alpha, beta

# EOF

# %%
