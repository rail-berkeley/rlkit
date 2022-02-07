import torch
from torch import jit, nn
from torch.nn import functional as F

from rlkit.pythonplusplus import identity


class Mlp(jit.ScriptModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation=F.elu,
        output_activation=identity,
        hidden_init=torch.nn.init.xavier_uniform_,
        b_init_value=0.0,
        apply_embedding=False,
        embedding_dim=0,
        num_embeddings=0,
        embedding_slice=0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        fc = nn.Linear(input_size, hidden_sizes[0])
        hidden_init(fc.weight)
        fc.bias.data.fill_(b_init_value)
        self.fc_block_1 = nn.Sequential(fc, hidden_activation(inplace=True))

        fc_block_2 = []
        in_size = hidden_sizes[0]

        for i, next_size in enumerate(hidden_sizes[1:]):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            fc_block_2.append(fc)
            fc_block_2.append(hidden_activation(inplace=True))

        last_fc = nn.Linear(in_size, output_size)
        hidden_init(last_fc.weight)
        last_fc.bias.data.fill_(0)
        fc_block_2.append(last_fc)

        self.fc_block_2 = nn.Sequential(*fc_block_2)
        self.apply_embedding = apply_embedding
        self.embedding_slice = embedding_slice
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    @jit.script_method
    def forward(self, input_):
        h = input_
        if self.apply_embedding:
            embed_h = h[:, : self.embedding_slice]
            embedding = self.embedding(embed_h.argmax(dim=1))
            h = torch.cat([embedding, h[:, self.embedding_slice :]], dim=1)
        h = self.fc_block_1(h)
        preactivation = self.fc_block_2(h)
        output = preactivation
        return output


class MlpResidual(Mlp):
    @jit.script_method
    def forward(self, input_):
        h = input_
        h_prev = h
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h, inplace=True)
            if i % 2 == 1:
                h = h + h_prev
            h_prev = h
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output
