import torch
from torch import nn
from lead.tfv6.diffusion_modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from lead.tfv6.diffusion_modules.conditional_unet1d import SinusoidalPosEmb
import copy
from enum import IntEnum
import numpy as np
from typing import Tuple, List, Optional, Dict, Union

# =========================================
DEFAULT_BETA_D = 2.0
DEFAULT_BETA_MIN = 0.1
DEFAULT_T = 1.0
T_NORMALIZE = 1000.0
DIFF_LOSS_WEIGHT = 2.0
DEFAULT_EGO_FUT_MODE = 20
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_BEV_DIMS = 1512
DEFAULT_EMBED_DIMS = 256
DEFAULT_TIME_EMBED_DIMS = 256

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    
    Args:
        x: Input tensor
        target_dims: Target number of dimensions
        
    Returns:
        Tensor with target_dims dimensions
        
    Raises:
        ValueError: If target_dims is less than input tensor's dimensions
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    for _ in range(dims_to_append):
        x = x.unsqueeze(-1)
    return x

class DDBMScheduler:
    """DDBM (Denoising Diffusion Bridging Model) scheduler for diffusion process."""
    
    def __init__(self, beta_d: float = DEFAULT_BETA_D, beta_min: float = DEFAULT_BETA_MIN, T: float = DEFAULT_T):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.T = T

    def vp_logs(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Calculate VP logs for given timestep."""
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def vp_logsnr(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Calculate VP logSNR for given timestep."""
        t = torch.as_tensor(t)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def get_abc(self, t: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate a_t, b_t, c_t coefficients for given timestep."""
        logsnr_t = self.vp_logsnr(t)
        logsnr_T = self.vp_logsnr(self.T)
        logs_t = self.vp_logs(t)
        logs_T = self.vp_logs(self.T)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()

        return a_t, b_t, c_t

    def add_noise(self, t: torch.Tensor, x0: torch.Tensor, xT: torch.Tensor, noise: torch.Tensor, 
                  t_normalize: float = T_NORMALIZE) -> torch.Tensor:
        """Add noise to the input tensor according to DDBM schedule.
        
        Args:
            t: Timestep tensor
            x0: Clean input tensor
            xT: Prior tensor
            noise: Noise tensor
            t_normalize: Normalization factor for timestep
            
        Returns:
            Noisy tensor sample
        """
        t = append_dims(t, x0.ndim) / t_normalize
        a_t, b_t, c_t = self.get_abc(t)
        samples = a_t * xT + b_t * x0 + c_t * noise

        return samples

    def sample_step(self, t: torch.Tensor, t_prev: torch.Tensor, xt: torch.Tensor, 
                    x0: torch.Tensor, xT: torch.Tensor, t_normalize: float = T_NORMALIZE) -> torch.Tensor:
        """Perform one sampling step of the DDBM process.
        
        Args:
            t: Current timestep
            t_prev: Previous timestep
            xt: Current tensor sample
            x0: Denoised tensor (from model)
            xT: Prior tensor
            t_normalize: Normalization factor for timestep
            
        Returns:
            Tensor sample at previous timestep
        """
        is_T = ((t / t_normalize) == self.T).all()
        t = append_dims(t, x0.ndim) / t_normalize
        t_prev = append_dims(t_prev, x0.ndim) / t_normalize

        a_t, b_t, c_t = self.get_abc(t)
        a_t_prev, b_t_prev, c_t_prev = self.get_abc(t_prev)

        xt_prev = a_t_prev * xT + b_t_prev * x0 
        if is_T:
            xt_prev = xt_prev + c_t_prev * torch.randn_like(xt_prev)
        else:
            xt_prev = xt_prev + (c_t_prev / c_t) * (xt - a_t * xT - b_t * x0)

        return xt_prev
    
class TrajectoryHead(nn.Module):
    """Trajectory prediction head with diffusion-based refinement."""
    
    def __init__(self, 
                 num_poses: int, 
                 d_ffn: int, 
                 d_model: int, 
                 plan_anchor_path: str,
                 config: Dict):
        """
        Initializes trajectory head.
        
        Args:
            num_poses: Number of (x,y,θ) poses to predict
            d_ffn: Dimensionality of feed-forward network
            d_model: Input dimensionality
            plan_anchor_path: Path to plan anchor numpy file
            config: Configuration dictionary
        """
        super().__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = DIFF_LOSS_WEIGHT
        self.ego_fut_mode = DEFAULT_EGO_FUT_MODE
        self.config = copy.deepcopy(config)

        # Mean and std initialization
        if self.config.diffusion_speed:
            self.config.speed_holder = 1
            self.x_mean = [2.490868595127002, 3.481156543917186, 4.464454848382859, 5.441355354234655, 6.415066909825394, 7.387900003104902, 8.359328053513595, 9.327316075722655, 10.290424566869847, 11.247967869978408, 4.702808110999116]
            self.x_std = [0.08869524498127068, 0.11285663272498085, 0.1441065214135421, 0.1931662097428255, 0.2527258929766773, 0.3145910978556985, 0.38050558536760937, 0.456217660900088, 0.5447450516202552, 0.646182474287767, 4.15365410738134]
            self.y_mean = [-0.012665788951539188, -0.024444082488842975, -0.047354389746646575, -0.07508421073500589, -0.10298006960418869, -0.1273045562122212, -0.14644950739563206, -0.16248700086691703, -0.17780588942014083, -0.19377927505995685, 0]
            self.y_std = [0.19376886519246483, 0.29851686549571804, 0.43366409659231814, 0.6016148827569083, 0.7882027369161408, 0.978601353702765, 1.1677537291437756, 1.3619226368298625, 1.56701254770298, 1.7858588878228894, 1.0]
        else:
            self.config.speed_holder = 0
            self.x_mean = [2.490868595127002, 3.481156543917186, 4.464454848382859, 5.441355354234655, 6.415066909825394, 7.387900003104902, 8.359328053513595, 9.327316075722655, 10.290424566869847, 11.247967869978408]
            self.x_std = [0.08869524498127068, 0.11285663272498085, 0.1441065214135421, 0.1931662097428255, 0.2527258929766773, 0.3145910978556985, 0.38050558536760937, 0.456217660900088, 0.5447450516202552, 0.646182474287767]
            self.y_mean = [-0.012665788951539188, -0.024444082488842975, -0.047354389746646575, -0.07508421073500589, -0.10298006960418869, -0.1273045562122212, -0.14644950739563206, -0.16248700086691703, -0.17780588942014083, -0.19377927505995685]
            self.y_std = [0.19376886519246483, 0.29851686549571804, 0.43366409659231814, 0.6016148827569083, 0.7882027369161408, 0.978601353702765, 1.1677537291437756, 1.3619226368298625, 1.56701254770298, 1.7858588878228894]

        # Convert to tensors (will be moved to correct device during forward pass)
        self.mean = torch.tensor([self.x_mean, self.y_mean]).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        self.std = torch.tensor([self.x_std, self.y_std]).transpose(0, 1).unsqueeze(0).unsqueeze(0)

        # Diffusion scheduler
        self.diffusion_scheduler = DDBMScheduler(
            beta_d=DEFAULT_BETA_D,
            beta_min=DEFAULT_BETA_MIN,
            T=DEFAULT_T,
        )

        # Load plan anchor
        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) 
        
        # Networks
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 64*(num_poses+self.config.speed_holder)), 
            nn.Linear(d_model, d_model),
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Decoder layers
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=self.config,
        )

        diff_decoder_layer_cls = CustomTransformerDecoderLayerCls(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=self.config,
        )

        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.diff_decoder_cls = CustomTransformerDecoderCls(diff_decoder_layer_cls, 2)

    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """Normalize odometry information using precomputed mean and std."""
        return (odo_info_fut - self.mean.to(odo_info_fut.device)) / self.std.to(odo_info_fut.device)

    def denorm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """Denormalize odometry information using precomputed mean and std."""
        return odo_info_fut * self.std.to(odo_info_fut.device) + self.mean.to(odo_info_fut.device)

    def forward(self, 
                ego_query: torch.Tensor, 
                bev_feature: torch.Tensor,
                bev_spatial_shape: Tuple[int, int], 
                targets: Optional[torch.Tensor] = None,
                global_img: Optional[torch.Tensor] = None, 
                gt_checkpoints: Optional[torch.Tensor] = None) -> Union[Dict, torch.Tensor]:
        """Forward pass of the trajectory head.
        
        Args:
            ego_query: Ego vehicle query tensor
            bev_feature: BEV feature tensor
            bev_spatial_shape: Spatial shape of BEV features
            targets: Target trajectories (unused)
            global_img: Global image features
            gt_checkpoints: Ground truth checkpoints (for training)
            
        Returns:
            Training: Dictionary with prediction outputs
            Evaluation: Predicted trajectory tensor
        """
        if self.training:
            return self.forward_train(ego_query, bev_feature, bev_spatial_shape, global_img=global_img, gt_checkpoints=gt_checkpoints)
        else:
            return self.forward_test(ego_query, bev_feature, bev_spatial_shape, global_img=global_img)

    def forward_train(self, 
                      ego_query: torch.Tensor, 
                      bev_feature: torch.Tensor, 
                      bev_spatial_shape: Tuple[int, int], 
                      global_img: Optional[torch.Tensor] = None, 
                      gt_checkpoints: torch.Tensor = None) -> Dict:
        """Forward pass for training mode."""
        bs = ego_query.shape[0]
        device = ego_query.device

        # Prepare plan anchor and ground truth
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0 = gt_checkpoints.unsqueeze(1)[...,:2].repeat(1, plan_anchor.shape[1], 1, 1)

        # Normalize and add noise
        odo_info_fut = self.norm_odo(x_0)
        odo_plan_anchor = self.norm_odo(plan_anchor)
        timesteps = torch.randint(1, 1001, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)

        noisy_traj_points = self.diffusion_scheduler.add_noise(
            x0=odo_info_fut,
            xT=odo_plan_anchor,
            noise=noise,
            t=timesteps,
        ).float()
        noisy_traj_points = self.denorm_odo(noisy_traj_points)
        
        if self.config.diffusion_speed:
            noisy_traj_points[...,-1,-1] = 0

        # Encode trajectory features
        ego_fut_mode = noisy_traj_points.shape[1]
        ak_ego_fut_mode = plan_anchor.shape[1]

        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)

        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs, ak_ego_fut_mode, -1).to(dtype=torch.float32)

        # Decode predictions
        poses_reg_list, _ = self.diff_decoder(
            traj_feature, noisy_traj_points, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query, time_embed, global_img
        )
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query
        )

        # Select best prediction
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses+self.config.speed_holder, 2)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        
        if self.config.diffusion_speed:
            best_reg[...,-1,-1] = 0

        return {"poses_reg_list": None, "poses_cls_list": poses_cls_list, "best_reg": best_reg}

    def forward_test(self, 
                     ego_query: torch.Tensor,
                     bev_feature: torch.Tensor,
                     bev_spatial_shape: Tuple[int, int],
                     global_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for evaluation mode (sampling)."""
        device = ego_query.device
        bs = ego_query.shape[0]
        
        step_num = self.config.step_num
        ts = torch.arange(0, 1001, 1000//step_num).to(device)

        # Prepare plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)

        # Encode anchor features
        ak_ego_fut_mode = plan_anchor.shape[1]
        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs, ak_ego_fut_mode, -1)

        # Get classification predictions
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query
        )
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses+self.config.speed_holder, 2)

        # Diffusion sampling loop
        xt_prev = plan_anchor
        for i in range(step_num, 0, -1):
            xt = xt_prev
            t = ts[i] * torch.ones([bs], device=device)
            t_prev = ts[i-1] * torch.ones([bs], device=device)

            # Encode current trajectory
            traj_pos_embed = gen_sineembed_for_position(xt, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, xt.shape[1], -1)

            time_embed = self.time_mlp(t)
            time_embed = time_embed.view(bs, 1, -1)

            # Get denoised prediction
            poses_reg_list, _ = self.diff_decoder(
                traj_feature, xt, plan_anchor, bev_feature, 
                bev_spatial_shape, ego_query, time_embed, global_img
            )

            x0_mean = poses_reg_list[-1]
            x0_mean[...,-1,-1] = 0

            # Sample previous timestep
            xt_prev_norm = self.diffusion_scheduler.sample_step(
                t=t,
                t_prev=t_prev,
                xt=self.norm_odo(xt.clone()),
                x0=self.norm_odo(x0_mean),
                xT=self.norm_odo(plan_anchor.clone()),
            )

            xt_prev = self.denorm_odo(xt_prev_norm)
            xt_prev[...,-1,-1] = 0

        # Select best mode
        x0_sample = torch.gather(xt_prev, 1, mode_idx).squeeze(1)

        return x0_sample


class DiffMotionPlanningRefinementModule(nn.Module):
    """Refinement module for motion planning with regression and classification branches."""
    
    def __init__(
        self,
        embed_dims: int = DEFAULT_EMBED_DIMS,
        ego_fut_ts: int = 8,
        ego_fut_mode: int = DEFAULT_EGO_FUT_MODE,
        if_zeroinit_reg: bool = False, 
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.if_zeroinit_reg = if_zeroinit_reg
        
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
        
    def forward(self, traj_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the refinement module."""
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls

class DiffMotionPlanningRefinementModuleCls(nn.Module):
    """Classification-only refinement module for motion planning."""
    
    def __init__(
        self,
        embed_dims: int = DEFAULT_EMBED_DIMS,
        ego_fut_ts: int = 8,
        ego_fut_mode: int = DEFAULT_EGO_FUT_MODE,
        if_zeroinit_reg: bool = False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.if_zeroinit_reg = if_zeroinit_reg

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification module."""
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)

        return plan_cls

class ModulationLayer(nn.Module):
    """Modulation layer for conditioning features with time and global information."""

    def __init__(self, embed_dims: int, condition_dims: int = DEFAULT_TIME_EMBED_DIMS):
        super().__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )
        
        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature: torch.Tensor,
        time_embed: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        global_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the modulation layer."""
        # Combine conditioning features
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], dim=-1)
        else:
            global_feature = time_embed
            
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], dim=-1)
        
        # Compute scale and shift
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Apply modulation
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer for trajectory prediction."""
    
    def __init__(self, 
                 num_poses: int,
                 d_model: int,
                 d_ffn: int,
                 config: Dict,
                 ):
        super().__init__()

        self.config = config
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout_T = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1_T = nn.Dropout(DEFAULT_DROPOUT_RATE)

        # Cross BEV attention layers
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder,
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS,
        )
        
        self.cross_bev_attention_T = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder, 
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS,
        )

        # Cross ego attention layers
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        self.cross_ego_attention_T = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        
        self.ffn_T = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.norm1_T = nn.LayerNorm(config.tf_d_model)
        self.norm2_T = nn.LayerNorm(config.tf_d_model)
        self.norm3_T = nn.LayerNorm(config.tf_d_model)

        # Modulation and task decoder
        self.time_modulation = ModulationLayer(config.tf_d_model * 2, DEFAULT_TIME_EMBED_DIMS)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model * 2,
            ego_fut_ts=num_poses + self.config.speed_holder, 
            ego_fut_mode=DEFAULT_EGO_FUT_MODE,
        )

    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,  
                plan_anchor: torch.Tensor,
                bev_feature: torch.Tensor,  
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor, 
                time_embed: Optional[torch.Tensor] = None, 
                global_img: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer decoder layer."""
        # Cross BEV attention
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature_T = self.cross_bev_attention_T(traj_feature, plan_anchor, bev_feature, bev_spatial_shape)
        
        traj_feature = self.norm1(traj_feature)
        traj_feature_T = self.norm1_T(traj_feature_T)

        # Cross attention with ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        traj_feature_T = traj_feature_T + self.dropout1_T(self.cross_ego_attention_T(traj_feature_T, ego_query, ego_query)[0])
        traj_feature_T = self.norm2_T(traj_feature_T)

        # Feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature_T = self.norm3_T(self.ffn_T(traj_feature_T))
        
        # Concatenate features
        traj_feature = torch.cat([traj_feature, traj_feature_T], dim=-1)
        
        # Time modulation
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)
        
        # Predict waypoints
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points

        return poses_reg[...,:2], poses_cls

class CustomTransformerDecoderLayerCls(nn.Module):
    """Classification-only transformer decoder layer."""
    
    def __init__(self, 
                 num_poses: int,
                 d_model: int,
                 d_ffn: int,
                 config: Dict,
                 ):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
        
        # Cross BEV attention
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder, 
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS, 
        )
        
        # Cross attention layers
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Modulation and task decoder
        self.time_modulation = ModulationLayer(d_model, DEFAULT_TIME_EMBED_DIMS)
        self.task_decoder_cls = DiffMotionPlanningRefinementModuleCls(
            embed_dims=d_model,
            ego_fut_ts=num_poses + self.config.speed_holder, 
            ego_fut_mode=DEFAULT_EGO_FUT_MODE,
        )

    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,  
                bev_feature: torch.Tensor,  
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification decoder layer."""
        # Cross BEV attention
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = self.norm1(traj_feature)
        
        # Cross attention with ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # Feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        
        # Predict classification
        poses_cls = self.task_decoder_cls(traj_feature)

        return poses_cls

class CustomTransformerDecoder(nn.Module):
    """Stack of custom transformer decoder layers for trajectory prediction."""
    
    def __init__(
        self, 
        decoder_layer: nn.Module, 
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,
                plan_anchor: torch.Tensor,
                bev_feature: torch.Tensor, 
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor, 
                time_embed: torch.Tensor, 
                global_img: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the transformer decoder stack."""
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature, traj_points, plan_anchor, bev_feature, 
                bev_spatial_shape, ego_query, time_embed, global_img
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
            
        return poses_reg_list, poses_cls_list

class CustomTransformerDecoderCls(nn.Module):
    """Stack of classification-only transformer decoder layers."""
    
    def __init__(
        self, 
        decoder_layer: nn.Module, 
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor, 
                bev_feature: torch.Tensor, 
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the classification decoder stack."""
        poses_cls_list = []
        traj_points = noisy_traj_points
        
        for mod in self.layers:
            poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, ego_query)
            poses_cls_list.append(poses_cls)
            
        return poses_cls_list

# =========================================
class BaseIndex(IntEnum):
    """Base class for index enums with common functionality."""
    
    @classmethod
    def size(cls) -> int:
        """Get number of valid indices (excluding private/special attributes)."""
        return len([attr for attr in cls.__members__.keys() if not attr.startswith('_')])

class StateSE2Index(BaseIndex):
    """Intenum for SE(2) state arrays (X, Y, Heading)."""
    X = 0
    Y = 1
    HEADING = 2

    @classmethod
    @property
    def POINT(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls) -> slice:
        """Slice for full SE(2) state (X, Y, HEADING)."""
        return slice(cls.X, cls.HEADING + 1)

class BoundingBoxIndex(BaseIndex):
    """Intenum for bounding box arrays."""
    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    HEADING = 6

    @classmethod
    @property
    def POINT2D(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def POSITION(cls) -> slice:
        """Slice for 3D position (X, Y, Z)."""
        return slice(cls.X, cls.Z + 1)

    @classmethod
    @property
    def DIMENSION(cls) -> slice:
        """Slice for dimensions (LENGTH, WIDTH, HEIGHT)."""
        return slice(cls.LENGTH, cls.HEIGHT + 1)

class LidarIndex(BaseIndex):
    """Intenum for lidar point cloud arrays."""
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
    ID = 5

    @classmethod
    @property
    def POINT2D(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def POSITION(cls) -> slice:
        """Slice for 3D position (X, Y, Z)."""
        return slice(cls.X, cls.Z + 1)
