import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    class Block(nn.Module):

        def __init__(self, d_in, d_out, bias, dropout):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.dropout(self.activation(self.linear(x)))

    def __init__(self,
                 d_in,
                 d_layers,
                 dropouts,
                 d_out
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )

        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):
        x = x.float().to(self.device)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        return x.squeeze()


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class MLPDiffusion(nn.Module):
    def __init__(self,
                 d_in,
                 num_classes,
                 d_layers,
                 dropouts,
                 is_y_cond=True,
                 dim_t=32
                 ):
        super().__init__()

        self.is_y_cond = is_y_cond
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_t = dim_t
        self.num_classes = num_classes

        self.mlp = MLP(d_in=dim_t, d_out=d_in, d_layers=d_layers, dropouts=dropouts)

        if self.num_classes > 0 and is_y_cond:
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond:
            self.label_emb = nn.Linear(1, dim_t)

        self.proj = nn.Linear(d_in, dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps, y=None):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:
                y = y.long().squeeze().to(self.device)
            else:
                y = y.resize(y.size(0), 1).float().to(self.device)
            emb += F.silu(self.label_emb(y))

        x = self.proj(x) + emb

        return self.mlp(x)
