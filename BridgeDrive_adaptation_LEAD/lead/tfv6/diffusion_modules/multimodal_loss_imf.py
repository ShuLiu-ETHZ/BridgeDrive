import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Optional, Tuple
from torch import Tensor
import math

def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class iMeanFlow(nn.Module):
    """
    PyTorch implementation of improved MeanFlow with v-loss.
    
    Args:
        model: The neural network model (e.g., transformer-based).
        num_classes: Number of classes for conditional generation.
        P_mean: Mean of logit-normal distribution for time sampling.
        P_std: Standard deviation of logit-normal distribution.
        data_proportion: Proportion of flow matching samples in batch.
        cfg_beta: Beta parameter for CFG scale sampling.
        class_dropout_prob: Probability of dropping class labels for CFG.
        norm_p: Power for adaptive weighting.
        norm_eps: Epsilon for adaptive weighting.
        dtype: Data type for computations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 1000,
        P_mean: float = -0.4,
        P_std: float = 1.0,
        data_proportion: float = 0.5,
        cfg_beta: float = 1.0,
        class_dropout_prob: float = 0.1,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.dtype = dtype
    
    #######################################################
    #                       Solver                        #
    #######################################################
    
    @torch.no_grad()
    def sample_one_step(
        self,
        z_t: Tensor,
        labels: Tensor,
        i: int,
        t_steps: Tensor,
        omega: float,
        t_min: float,
        t_max: float,
    ) -> Tensor:
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t, shape (B, C, H, W).
            labels: Class labels for the batch, shape (B,).
            i: Current time step index.
            t_steps: Array of time steps, shape (num_steps + 1,).
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            
        Returns:
            Updated z_t at time step r.
        """
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]
        
        t = t.expand(bsz)
        r = r.expand(bsz)
        omega = torch.tensor(omega, device=z_t.device).expand(bsz)
        t_min = torch.tensor(t_min, device=z_t.device).expand(bsz)
        t_max = torch.tensor(t_max, device=z_t.device).expand(bsz)
        
        u = self.u_fn(z_t, r, t, omega, t_min, t_max, y=labels)
        
        # z_{r} = z_t - (t - r) * u
        delta_t = (t - r).view(-1, 1, 1, 1)
        return z_t - delta_t * u
    
    @torch.no_grad()
    def generate(
        self,
        n_sample: int,
        img_size: int,
        img_channels: int,
        num_steps: int,
        omega: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        sample_idx: Optional[int] = None,
        device: str = 'cuda',
    ) -> Tensor:
        """
        Generate samples from the model.
        
        Args:
            n_sample: Number of samples to generate.
            img_size: Image size.
            img_channels: Number of image channels.
            num_steps: Number of sampling steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            sample_idx: Optional index for class-conditional sampling.
            device: Device to run on.

        Returns:
            images: Generated images, shape (n_sample, C, H, W).
        """
        x_shape = (n_sample, img_channels, img_size, img_size)
        z_t = torch.randn(x_shape, dtype=self.dtype, device=device)
        
        if sample_idx is not None:
            all_y = torch.arange(n_sample, dtype=torch.int64, device=device)
            y = (all_y + sample_idx * n_sample) % self.num_classes
        else:
            y = torch.randint(0, self.num_classes, (n_sample,), device=device)
        
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            z_t = self.sample_one_step(z_t, y, i, t_steps, omega, t_min, t_max)
        
        return z_t
    
    #######################################################
    #                       Schedule                      #
    #######################################################
    
    def logit_normal_dist(self, bz: int, device: str) -> Tensor:
        """Sample from logit-normal distribution."""
        rnd_normal = torch.randn(bz, 1, 1, 1, dtype=self.dtype, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)
    
    def sample_tr(self, bz: int, device: str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample t and r from logit-normal distribution.
        
        Returns:
            t: Time step t, shape (bz, 1, 1, 1).
            r: Time step r, shape (bz, 1, 1, 1).
            fm_mask: Flow matching mask, shape (bz, 1, 1, 1).
        """
        t = self.logit_normal_dist(bz, device)
        r = self.logit_normal_dist(bz, device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        
        data_size = int(bz * self.data_proportion)
        fm_mask = torch.arange(bz, device=device)[:, None, None, None] < data_size
        
        # For flow matching samples, set r = t
        r = torch.where(fm_mask, t, r)
        
        return t, r, fm_mask
    
    def sample_cfg_scale(self, bz: int, device: str, s_max: float = 7.0) -> Tensor:
        """
        Sample CFG scale omega from power distribution.
        
        Returns:
            s: CFG scale, shape (bz, 1, 1, 1).
        """
        u = torch.rand(bz, 1, 1, 1, device=device, dtype=torch.float32)
        
        if self.cfg_beta == 1.0:
            s = torch.exp(u * math.log1p(s_max))
        else:
            log_base = (1.0 - self.cfg_beta) * math.log1p(s_max)
            log_inner = torch.log1p(u * (math.exp(log_base) - 1))
            s = torch.exp(log_inner / (1.0 - self.cfg_beta))
        
        return s.to(self.dtype)
    
    def sample_cfg_interval(
        self, bz: int, device: str, fm_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        
        Returns:
            t_min: Lower bound of guidance interval, shape (bz, 1, 1, 1).
            t_max: Upper bound of guidance interval, shape (bz, 1, 1, 1).
        """
        t_min = torch.rand(bz, 1, 1, 1, device=device, dtype=self.dtype) * 0.5
        t_max = torch.rand(bz, 1, 1, 1, device=device, dtype=self.dtype) * 0.5 + 0.5
        
        if fm_mask is not None:
            t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
            t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        
        return t_min, t_max
    
    #######################################################
    #               Training Utils & Guidance             #
    #######################################################
    
    def u_fn(
        self,
        z: Tensor,
        r: Tensor,
        t: Tensor,
        omega: Tensor,
        t_min: Tensor,
        t_max: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Compute the average velocity from x-prediction.

        Args:
            z: Noisy image at time t, shape (B, C, H, W).
            r: Time step r, shape (B,).
            t: Current time step, shape (B,).
            omega: CFG scale, shape (B,).
            t_min, t_max: Guidance interval, shape (B,).
            y: Class labels, shape (B,).
            
        Returns:
            u: Predicted u (average velocity field), shape (B, C, H, W).
        """
        # Average velocity: u = (z - x_pred) / t
        x_pred = self.model(z, r, t, omega, t_min, t_max, y)
        return (z - x_pred) / t.view(-1, 1, 1, 1)
    
    def v_cond_fn(self, z: Tensor, t: Tensor, omega: Tensor, y: Tensor) -> Tensor:
        """
        Compute the predicted v component (instantaneous velocity) conditioned on class labels.
        v = u_fn(z, t, t) - instantaneous velocity at time t

        Args:
            z: Noisy image at time t, shape (B, C, H, W).
            t: Current time step, shape (B,).
            omega: CFG scale, shape (B,).
            y: Class labels, shape (B,).
        
        Returns:
            v: Predicted v component, shape (B, C, H, W).
        """
        # Set t_min, t_max to dummy values for v prediction
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        
        # Instantaneous velocity: v = u_fn(z, t, t)
        v = self.u_fn(z, t, t, omega, t_min, t_max, y=y)
        
        return v
    
    def v_fn(
        self, x: Tensor, t: Tensor, omega: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute both conditioned and unconditioned predicted v components.

        Args:
            x: Noisy image at time t, shape (B, C, H, W).
            t: Current time step, shape (B,).
            omega: CFG scale, shape (B,).
            y: Class labels, shape (B,).

        Returns:
            v_c: Predicted v component conditioned on class labels, shape (B, C, H, W).
            v_u: Predicted v component without class labels, shape (B, C, H, W).
        """
        bz = x.shape[0]
        
        # Create duplicated batch for conditioned and unconditioned predictions
        x = torch.cat([x, x], dim=0)
        y_null = torch.full((bz,), self.num_classes, dtype=y.dtype, device=y.device)
        y = torch.cat([y, y_null], dim=0)
        t = torch.cat([t, t], dim=0)
        w = torch.cat([omega, torch.ones_like(omega)], dim=0)
        
        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = torch.chunk(out, 2, dim=0)
        
        return v_c, v_u
    
    def cond_drop(
        self, v_t: Tensor, v_g: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t, shape (B, C, H, W).
            v_g: Guided instantaneous velocity at time t, shape (B, C, H, W).
            labels: Class labels for the batch, shape (B,).

        Returns:
            labels: Possibly dropped class labels, shape (B,).
            v_g: Modified guided instantaneous velocity at time t, shape (B, C, H, W).
                 For samples with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]
        
        rand_mask = torch.rand(bz, device=labels.device) < self.class_dropout_prob
        num_drop = rand_mask.sum().item()
        drop_mask = torch.arange(bz, device=labels.device).unsqueeze(1).unsqueeze(2).unsqueeze(3) < num_drop
        
        labels = torch.where(
            drop_mask.squeeze(),
            torch.full_like(labels, self.num_classes),
            labels,
        )
        v_g = torch.where(drop_mask, v_t, v_g)
        
        return labels, v_g
    
    def guidance_fn(
        self,
        v_t: Tensor,
        z_t: Tensor,
        t: Tensor,
        r: Tensor,
        y: Tensor,
        fm_mask: Tensor,
        w: Tensor,
        t_min: Tensor,
        t_max: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t, shape (B, C, H, W).
            z_t: Noisy image at time t, shape (B, C, H, W).
            t, r: Two time steps, shape (B, 1, 1, 1).
            y: Class labels, shape (B,).
            fm_mask: Mask for t=r samples, shape (B, 1, 1, 1).
            t_min, t_max: Guidance interval, shape (B, 1, 1, 1).
            w: CFG scale, shape (B, 1, 1, 1).

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training, shape (B, C, H, W).
            v_c: Conditioned instantaneous velocity at time t, for jvp computation, shape (B, C, H, W).
        """
        # Compute CFG target
        t_flat = t.squeeze()
        w_flat = w.squeeze()
        
        v_c, v_u = self.v_fn(z_t, t_flat, w_flat, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)
        
        # Apply guidance interval
        in_interval = (t >= t_min) & (t <= t_max)
        w_adjusted = torch.where(in_interval, w, torch.ones_like(w))
        
        v_c = self.v_cond_fn(z_t, t_flat, w_adjusted.squeeze(), y=y)
        v_g = v_t + (1 - 1 / w_adjusted) * (v_c - v_u)
        
        # For flow matching samples, there is no CFG interval
        v_g = torch.where(fm_mask, v_g_fm, v_g)
        
        return v_g, v_c
    
    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    
    def forward(
        self, images: Tensor, labels: Tensor
    ) -> Tuple[Tensor, dict]:
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, C, H, W).
            labels: Corresponding class labels, shape (B,).
        
        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.to(self.dtype)
        bsz = images.shape[0]
        device = images.device
        
        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bsz, device)
        
        e = torch.randn_like(x, dtype=self.dtype)
        z_t = (1 - t) * x + t * e
        v_t = e - x
        
        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bsz, device, fm_mask)
        omega = self.sample_cfg_scale(bsz, device)
        
        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
        )
        
        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)
        
        # Enable gradient computation for JVP
        z_t_jvp = z_t.detach().requires_grad_(True)
        r_flat = r.squeeze().detach().requires_grad_(True)
        t_flat = t.squeeze().detach().requires_grad_(True)
        
        # Instantaneous velocity: v = u_fn(z, t, t)
        v = self.u_fn(
            z_t_jvp, t_flat, t_flat, omega.squeeze(), 
            t_min.squeeze(), t_max.squeeze(), y=labels
        )
        
        # Compute u and du/dt using JVP in one autograd call
        # jvp(u_fn, (z, r, t), (v, 0, 1))
        # This computes: u = u_fn(z, r, t) and du/dt = ∂u/∂z * v + ∂u/∂r * 0 + ∂u/∂t * 1
        u = self.u_fn(
            z_t_jvp, r_flat, t_flat, omega.squeeze(),
            t_min.squeeze(), t_max.squeeze(), y=labels
        )
        
        # JVP: compute total derivative du/dt with tangent vectors (v, 0, 1)
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=(z_t_jvp, r_flat, t_flat),
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )
        # du/dt = ∂u/∂z * v + ∂u/∂r * 0 + ∂u/∂t * 1
        du_dt_total = (du_dt[0] * v).sum(dim=(1, 2, 3), keepdim=True).expand_as(u) + du_dt[2].view(-1, 1, 1, 1)
        
        # Compound function V = u + (t - r) * stopgrad(du/dt)
        V = u + (t - r).view(-1, 1, 1, 1) * du_dt_total.detach()
        
        v_g = v_g.detach()
        
        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / adp_wt.detach()
        
        # improved MeanFlow objective: loss = metric(V, e - x)
        loss_value = torch.sum((V - v_g) ** 2, dim=(1, 2, 3))
        loss_value = adp_wt_fn(loss_value)
        
        loss = loss_value.mean()  # mean over batch
        
        dict_losses = {
            "loss": loss.item(),
            "loss_V": torch.mean((V - v_g) ** 2).item(),
        }
        
        return loss, dict_losses




