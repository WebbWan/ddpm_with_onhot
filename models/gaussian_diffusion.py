import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class GaussianDiffusion(nn.Module):

    def __init__(self,
                 *,
                 num_timesteps: int,
                 loss_type='huber',
                 scheduler='quad',
                 dtype=torch.float32,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()

        self.device = device
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.scheduler = scheduler

        betas = get_beta_schedule(
            scheduler,
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=num_timesteps
        )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = torch.tensor(betas, dtype=dtype, device=device)
        self.alphas = torch.tensor(alphas, dtype=dtype, device=device)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=dtype, device=device)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=dtype, device=device)

        # 计算扩散过程 q(x_t | x_{t-1}) 和其他
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=dtype, device=device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=dtype, device=device)
        self.log_one_minus_alphas_cumprod = torch.tensor(np.log(1. - alphas_cumprod), dtype=dtype, device=device)
        self.sqrt_recip_alphas_cumprod = torch.tensor(np.sqrt(1. / alphas_cumprod), dtype=dtype, device=device)
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(np.sqrt(1. / alphas_cumprod - 1), dtype=dtype, device=device)

        # 计算重建过程 q(x_{t-1} | x_t, x_0) 和其他
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=dtype, device=device)

        self.posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=dtype,
                                                           device=device)
        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
            dtype=dtype,
            device=device,
        )
        self.posterior_mean_coef2 = torch.tensor(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod),
            dtype=dtype,
            device=device
        )

    @staticmethod
    def _extract(a: torch.Tensor, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device).to(torch.int64)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumpord, t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, denoise_fn, x_start, t, noise=None):
        batch_size, attr_num = x_start.shape
        assert t.shape[0] == batch_size

        if noise is None:
            noise = torch.randn_like(x_start, dtype=x_start.dtype)

        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
        x_noisy = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise
        )

        x_recon = denoise_fn(x_noisy, t)
        if self.loss_type == 'huber':
            assert x_recon.shape == x_start.shape
            losses = F.smooth_l1_loss(noise, x_recon)
        elif self.loss_type == "l1":
            assert x_recon.shape == x_start.shape
            losses = F.l1_loss(noise, x_recon)
        elif self.loss_type == "l2":
            assert x_recon.shape == x_start.shape
            losses = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def p_mean_variance(self, denoise_fn, *, x, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, denoise_fn, *, x, t):
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t)
        noise = torch.randn_like(x, device=self.device)
        assert noise.shape == x.shape
        # 当 t == 0 不加噪
        nonzero_mask = torch.reshape(1 - torch.eq(t, 0).to(torch.float32),
                                     [x.shape[0]] + [1] * (len(x.shape) - 1)
                                     )
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, *, shape, noise_fn=torch.randn):
        assert isinstance(shape, (tuple, list))
        x_0 = noise_fn(size=shape, dtype=torch.float32, device=self.device)
        x_final = x_0
        for timestep in reversed(range(self.num_timesteps)):
            x_final = self.p_sample(denoise_fn=denoise_fn, x=x_final,
                                    t=torch.full([shape[0]], timestep, device=self.device))
        return x_final
