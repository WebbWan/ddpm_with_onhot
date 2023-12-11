import time

import numpy as np
import torch
import functools
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.mlp import MLPDiffusion
from models.gaussian_diffusion import GaussianDiffusion


class Model(nn.Module):
    def __init__(self,
                 *,
                 dim: int,
                 lr: float,
                 wd: float,
                 num_class: int,
                 d_layer: list,
                 dim_t=32,
                 epochs=1000,
                 batch_size=64,
                 num_timesteps=1000,
                 dropouts=0.0
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.lr = lr
        self.wd = wd
        self.num_class = num_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.d_layer = d_layer
        self.dropouts = dropouts
        self.diffusion = GaussianDiffusion(num_timesteps=num_timesteps).to(self.device)
        self.eps_model = MLPDiffusion(
            d_in=dim,
            num_classes=num_class,
            d_layers=d_layer,
            dropouts=dropouts,
            is_y_cond=True,
            dim_t=dim_t,
        ).to(self.device)

    def _denoise(self, x: torch.Tensor, t, y=None):
        batch_size, attr_size = x.shape
        x = x.to(torch.float32)
        assert x.dtype == torch.float32
        assert t.shape[0] == batch_size and t.dtype in [torch.int32, torch.int64]
        out = self.eps_model(x, t, y)
        return out

    def train_fn(self, x, optimizer, y=None):
        batch_size, attr_size = x.shape
        t = torch.rand(size=[batch_size], dtype=torch.float32, device=self.device) * \
            torch.full(size=[batch_size], fill_value=self.num_timesteps, device=self.device)
        t = t.to(torch.int32)
        optimizer.zero_grad()
        losses = self.diffusion.p_losses(
            denoise_fn=functools.partial(self._denoise, y=y), x_start=x, t=t
        )
        losses.backward()
        optimizer.step()
        return losses.cpu()

    @torch.no_grad()
    def samples_fn(self, sample_num, y=None):
        self.eps_model.eval()
        step = sample_num // self.batch_size + 1
        data = []
        progress_bar = tqdm(total=step, desc="Sampling Progress", position=0, leave=False)
        with torch.no_grad():
            for _ in range(step):
                x = self.diffusion.p_sample_loop(
                    denoise_fn=functools.partial(self._denoise, y=y),
                    shape=[self.batch_size, self.dim],
                    noise_fn=torch.randn
                ).cpu()
                data.append(x)
                progress_bar.set_postfix({"Step": _ + 1}, refresh=True)
                progress_bar.update(1)

        progress_bar.close()
        data = np.concatenate(data, axis=0)
        x_sample = data[:sample_num]
        return x_sample

    def train_loop(self, data: torch.Tensor, y=None):
        dataset = TensorDataset(data.to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size)

        optim = torch.optim.Adam(
            self.eps_model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )

        self.eps_model.train()
        time.sleep(0.05)
        progress_bar = tqdm(total=self.epochs, desc="Training Progress", position=0, leave=False)

        for epoch in range(self.epochs):
            loss_res = []
            for idx, data in enumerate(loader):
                loss = self.train_fn(data[0].float(), optim, y)
                loss_res.append(loss)

            epoch_loss_mean = torch.mean(torch.tensor(loss_res))
            progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": epoch_loss_mean.item()}, refresh=True)
            progress_bar.update(1)
        progress_bar.close()