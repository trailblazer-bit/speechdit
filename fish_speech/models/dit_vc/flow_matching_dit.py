# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import os
import math
from fish_speech.models.flow_decoder.basecfm import BASECFM

class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, cond, mu, t, mask.squeeze(1),spk_embeds=spks)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.estimator(
                    x, torch.zeros_like(cond) if cond is not None else None,
                    torch.zeros_like(mu), t,
                    mask = mask.squeeze(1),
                    spk_embeds = torch.zeros_like(spks) if spks is not None else None
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None
            spks = spks * cfg_mask.view(-1, 1, 1) if spks is not None else None
        pred = self.estimator(y,cond, mu, time = t.squeeze(), mask = mask.squeeze(1), spk_embeds=spks)  # (B,C,T)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

class ConditionalCFM_ShortCut_Wispk(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None,
                    DENOISE_TIMESTEPS=128,BOOTSTRAP_EVERY=8):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.DENOISE_TIMESTEPS = DENOISE_TIMESTEPS
        self.BOOTSTRAP_EVERY = BOOTSTRAP_EVERY

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond,n_timesteps=n_timesteps)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, n_timesteps=4):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dt_base = torch.ones_like(t).to(t.device) * math.log2(n_timesteps)
            dphi_dt = self.estimator(x, cond, mu, t, mask.squeeze(1),spk_embeds=spks,dt_base_bootstrap=dt_base)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.estimator(
                    x, torch.zeros_like(cond) if cond is not None else None,
                    torch.zeros_like(mu), t,
                    mask = mask.squeeze(1),
                    spk_embeds = torch.zeros_like(spks) if spks is not None else None,
                    dt_base_bootstrap=dt_base
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        x_t, v_t, t, dt_base, cond_dropped, mu_dropped, spks_dropped = self.create_targets(x1,mask,mu,spks,cond)
        pred = self.estimator(x_t,cond_dropped, mu_dropped, time = t, mask = mask.squeeze(1), dt_base_bootstrap=dt_base,spk_embeds=spks_dropped)  # (B,C,T)
        loss = F.mse_loss(pred * mask, v_t * mask, reduction="sum") / (torch.sum(mask) * v_t.shape[1])
        return loss, x_t

    def create_targets(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        FORCE_T = -1
        FORCE_DT = -1

        # 1. create step sizes dt
        bootstrap_batch_size = b // self.BOOTSTRAP_EVERY #=8
        log2_sections = int(math.log2(self.DENOISE_TIMESTEPS))

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)

        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)])

        force_dt_vec = torch.ones(bootstrap_batch_size) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(mu.device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(mu.device)
        
        t = t / dt_sections
        
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(mu.device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(mu.device)
        t_full = t[:, None, None]

        # 3. generate bootstrap targets:
        x_1 = x1[:bootstrap_batch_size]
        x_0 = torch.randn_like(x_1)
        cond_1 = cond[:bootstrap_batch_size]
        mu_1 = mu[:bootstrap_batch_size]
        spks_1 = spks[:bootstrap_batch_size]
        mask_1 = mask[:bootstrap_batch_size]

        # get dx at timestep t
        x_t = (1 - (1-self.sigma_min) * t_full)*x_0 + t_full*x_1

        with torch.no_grad():
            v_b1 = self.estimator(x_t, cond_1,mu_1, time=t, mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap,spk_embeds=spks_1)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
        
        with torch.no_grad():
            v_b2 = self.estimator(x_t2, cond_1,mu_1, time=t2,mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap, spk_embeds=spks_1)

        v_target = (v_b1 + v_b2) / 2
        
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t

        # 4. generate flow-matching targets
        # sample t(normalized)
        t = torch.randint(low=0, high=self.DENOISE_TIMESTEPS, size=(mu.shape[0],),device=mu.device, dtype=torch.float32)
        t /= self.DENOISE_TIMESTEPS
        # t = torch.rand([b,], device=mu.device, dtype=mu.dtype)
        t_full = t[:, None, None]

        # sample flow pairs x_t, v_t
        x_0 = torch.randn_like(x1).to(mu.device)
        x_1 = x1
        x_t = (1 - (1 - self.sigma_min) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - self.sigma_min) * x_0

        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        dt_base = (torch.ones((b,), dtype=torch.int32) * dt_flow).to(mu.device)

        # 5. merge flow and bootstrap
        bst_size = b // self.BOOTSTRAP_EVERY
        bst_size_data = b - bst_size

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)

        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu_ = mu * cfg_mask.view(-1, 1, 1)
            cond_ = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None
            spks_ = spks * cfg_mask.view(-1, 1, 1) if spks is not None else None
        cond_dropped = torch.cat([cond_1, cond_[:bst_size_data]], dim=0)
        mu_dropped = torch.cat([mu_1, mu_[:bst_size_data]], dim=0)
        spks_dropped = torch.cat([spks_1, spks_[:bst_size_data]], dim=0)

        return x_t, v_t, t, dt_base, cond_dropped, mu_dropped, spks_dropped


class ConditionalCFM_ShortCut_Wispk_Full(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None,
                    DENOISE_TIMESTEPS=128,BOOTSTRAP_EVERY=8,k=0.5):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.DENOISE_TIMESTEPS = DENOISE_TIMESTEPS
        self.BOOTSTRAP_EVERY = BOOTSTRAP_EVERY
        self.k = k

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond,n_timesteps=n_timesteps)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, n_timesteps=4):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dt_base = torch.ones_like(t).to(t.device) * math.log2(n_timesteps)
            dphi_dt = self.estimator(x, cond, mu, t, mask.squeeze(1),spk_embeds=spks,dt_base_bootstrap=dt_base)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.estimator(
                    x, torch.zeros_like(cond) if cond is not None else None,
                    torch.zeros_like(mu), t,
                    mask = mask.squeeze(1),
                    spk_embeds = torch.zeros_like(spks) if spks is not None else None,
                    dt_base_bootstrap=dt_base
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        x_t, v_t, t, dt_base, cond_dropped, mu_dropped, spks_dropped = self.create_targets(x1,mask,mu,spks,cond)
        pred = self.estimator(x_t,cond_dropped, mu_dropped, time = t, mask = mask.squeeze(1), dt_base_bootstrap=dt_base,spk_embeds=spks_dropped)  # (B,C,T)
        loss = F.mse_loss(pred * mask, v_t * mask, reduction="sum") / (torch.sum(mask) * v_t.shape[1])
        return loss, x_t

    def create_targets(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        FORCE_T = -1
        FORCE_DT = -1

        # 1. create step sizes dt
        # bootstrap_batch_size = b // self.BOOTSTRAP_EVERY #=8
        bootstrap_batch_size = int(b * self.k)
        log2_sections = int(math.log2(self.DENOISE_TIMESTEPS))

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)

        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)])

        force_dt_vec = torch.ones(bootstrap_batch_size) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(mu.device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(mu.device)
        
        t = t / dt_sections
        
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(mu.device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(mu.device)
        t_full = t[:, None, None]

        # 3. generate bootstrap targets:
        x_1 = x1[:bootstrap_batch_size]
        x_0 = torch.randn_like(x_1)
        cond_1 = cond[:bootstrap_batch_size]
        mu_1 = mu[:bootstrap_batch_size]
        spks_1 = spks[:bootstrap_batch_size]
        mask_1 = mask[:bootstrap_batch_size]

        # get dx at timestep t
        x_t = (1 - (1-self.sigma_min) * t_full)*x_0 + t_full*x_1

        with torch.no_grad():
            v_b1 = self.estimator(x_t, cond_1,mu_1, time=t, mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap,spk_embeds=spks_1)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
        
        with torch.no_grad():
            v_b2 = self.estimator(x_t2, cond_1,mu_1, time=t2,mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap, spk_embeds=spks_1)

        v_target = (v_b1 + v_b2) / 2
        
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t

        # 4. generate flow-matching targets
        # sample t(normalized)
        t = torch.randint(low=0, high=self.DENOISE_TIMESTEPS, size=(mu.shape[0],),device=mu.device, dtype=torch.float32)
        t /= self.DENOISE_TIMESTEPS
        # t = torch.rand([b,], device=mu.device, dtype=mu.dtype)
        t_full = t[:, None, None]

        # sample flow pairs x_t, v_t
        x_0 = torch.randn_like(x1).to(mu.device)
        x_1 = x1
        x_t = (1 - (1 - self.sigma_min) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - self.sigma_min) * x_0

        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        dt_base = (torch.ones((b,), dtype=torch.int32) * dt_flow).to(mu.device)

        # 5. merge flow and bootstrap
        # bst_size = b // self.BOOTSTRAP_EVERY
        bst_size = int(b * self.k)
        bst_size_data = b - bst_size

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)

        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu_ = mu * cfg_mask.view(-1, 1, 1)
            cond_ = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None
            spks_ = spks * cfg_mask.view(-1, 1, 1) if spks is not None else None
        cond_dropped = torch.cat([cond_1, cond_[:bst_size_data]], dim=0)
        mu_dropped = torch.cat([mu_1, mu_[:bst_size_data]], dim=0)
        spks_dropped = torch.cat([spks_1, spks_[:bst_size_data]], dim=0)

        return x_t, v_t, t, dt_base, cond_dropped, mu_dropped, spks_dropped

class ConditionalCFM_ShortCut_Wospk(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None,
                    DENOISE_TIMESTEPS=128,BOOTSTRAP_EVERY=8):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.DENOISE_TIMESTEPS = DENOISE_TIMESTEPS
        self.BOOTSTRAP_EVERY = BOOTSTRAP_EVERY

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond,n_timesteps=n_timesteps)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, n_timesteps=4):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dt_base = torch.ones_like(t).to(t.device) * math.log2(n_timesteps)
            dphi_dt = self.estimator(x, cond, mu, t, mask.squeeze(1),spk_embeds=spks,dt_base_bootstrap=dt_base)
            # Classifier-Free Guidance inference introduced in VoiceBox
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.estimator(
                    x, torch.zeros_like(cond) if cond is not None else None,
                    torch.zeros_like(mu), t,
                    mask = mask.squeeze(1),
                    spk_embeds = torch.zeros_like(spks) if spks is not None else None,
                    dt_base_bootstrap=dt_base
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        x_t, v_t, t, dt_base, cond_dropped, mu_dropped = self.create_targets(x1,mask,mu,cond=cond)
        pred = self.estimator(x_t,cond_dropped, mu_dropped, time = t, mask = mask.squeeze(1), dt_base_bootstrap=dt_base)  # (B,C,T)
        loss = F.mse_loss(pred * mask, v_t * mask, reduction="sum") / (torch.sum(mask) * v_t.shape[1])
        return loss, x_t

    def create_targets(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        FORCE_T = -1
        FORCE_DT = -1

        # 1. create step sizes dt
        bootstrap_batch_size = b // self.BOOTSTRAP_EVERY #=8
        log2_sections = int(math.log2(self.DENOISE_TIMESTEPS))

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)

        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)])

        force_dt_vec = torch.ones(bootstrap_batch_size) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(mu.device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(mu.device)
        
        t = t / dt_sections
        
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(mu.device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(mu.device)
        t_full = t[:, None, None]

        # 3. generate bootstrap targets:
        x_1 = x1[:bootstrap_batch_size]
        x_0 = torch.randn_like(x_1)
        cond_1 = cond[:bootstrap_batch_size]
        mu_1 = mu[:bootstrap_batch_size]
        mask_1 = mask[:bootstrap_batch_size]

        # get dx at timestep t
        x_t = (1 - (1-self.sigma_min) * t_full)*x_0 + t_full*x_1

        with torch.no_grad():
            v_b1 = self.estimator(x_t, cond_1,mu_1, time=t, mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1
        
        with torch.no_grad():
            v_b2 = self.estimator(x_t2, cond_1,mu_1, time=t2,mask=mask_1.squeeze(1),dt_base_bootstrap=dt_base_bootstrap)

        v_target = (v_b1 + v_b2) / 2
        
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t

        # 4. generate flow-matching targets
        # sample t(normalized)
        t = torch.rand([b,], device=mu.device, dtype=mu.dtype)
        t_full = t[:, None, None]

        # sample flow pairs x_t, v_t
        x_0 = torch.randn_like(x1).to(mu.device)
        x_1 = x1
        x_t = (1 - (1 - self.sigma_min) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - self.sigma_min) * x_0

        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        dt_base = (torch.ones((b,), dtype=torch.int32) * dt_flow).to(mu.device)

        # 5. merge flow and bootstrap
        bst_size = b // self.BOOTSTRAP_EVERY
        bst_size_data = b - bst_size

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)

        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu_ = mu * cfg_mask.view(-1, 1, 1)
            cond_ = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None
        cond_dropped = torch.cat([cond_1, cond_[:bst_size_data]], dim=0)
        mu_dropped = torch.cat([mu_1, mu_[:bst_size_data]], dim=0)

        return x_t, v_t, t, dt_base, cond_dropped, mu_dropped