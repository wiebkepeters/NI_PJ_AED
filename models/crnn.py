# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import zeros, Tensor
import torch
from torch.nn import Module, GRUCell, Linear
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from ._modules import DNN

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['CRNN']


class CRNN(Module):

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

        self.rnn_hh_size: int = rnn_out_dim
        self.nb_classes: int = nb_classes

        self.dnn: Module = DNN(
            cnn_channels=cnn_channels,
            cnn_dropout=cnn_dropout)

        self.rnn: Module = GRUCell(
            input_size=rnn_in_dim,
            hidden_size=self.rnn_hh_size,
            bias=True)

        self.classifier: Module = Linear(
            in_features=self.rnn_hh_size,
            out_features=self.nb_classes,
            bias=True)

    def forward(self,
                packed_input: Tensor, packed:bool) \
            -> Tensor:
        """Forward pass of the CRNN model.

        :param x: Input to the CRNN.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        # b_size, t_steps, _ = packed_input.data.size()

        packed_feats_data: Tensor = self.dnn(packed_input.data)#.permute(
            # 0, 2, 1, 3
        # ).reshape(b_size, t_steps, -1)


        packed_states = PackedSequence(
            packed_feats_data,
            packed_input.batch_sizes,
            packed_input.sorted_indices,
            packed_input.unsorted_indices)

        print(packed_states.data[0].shape)


        # h: Tensor = zeros(
        #     b_size, self.rnn_hh_size
        # ).to(packed_input.device)

        # packed_outputs: Tensor = zeros(
        #     b_size, t_steps, self.nb_classes
        # ).to(packed_input.device)

        # for i, t_step in enumerate(packed_states.data.permute(1, 0, 2)):
        #     h: Tensor = self.rnn(t_step, h)
        #     packed_outputs[:, i, :] = self.classifier(h)

        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # return outputs

# EOF

# %%
