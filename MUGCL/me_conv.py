from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
from torch import nn
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn.inits import reset

# from torch_geometric.nn import aggr

class MeConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels,input_size=768, hidden_size=768, **kwargs):
        super(MeConv, self).__init__(aggr='max', **kwargs)
        self.nn = Seq(Linear(in_channels*2, out_channels))
        self.reset_parameters()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        output, (hidden_state, cell_state) = self.lstm(x_j)
        #return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        return self.nn(torch.cat([x_i, output], dim=-1))
        # return self.nn(x_j-x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

