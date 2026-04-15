import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
from navsim.agents.diffusiondrive.modules.multimodal_loss import DDBMLossComputer


import copy
from enum import IntEnum

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

class DDBMScheduler:

    def __init__(self, beta_d=19.9, beta_min=0.1, T=1.0):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.T = T

    def vp_logs(self, t):
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * (self.beta_d) - 0.5 * t * self.beta_min

    def vp_logsnr(self, t):
        t = torch.as_tensor(t)
        return - torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def get_abc(self, t):

        logsnr_t = self.vp_logsnr(t)
        logsnr_T = self.vp_logsnr(self.T)
        logs_t = self.vp_logs(t)
        logs_T = self.vp_logs(self.T)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()

        return a_t, b_t, c_t

    def add_noise(self, t, x0, xT, noise, t_normalize = 1000.0):
        # x_t ~ q(x_t|x_0, x_T)
        t = append_dims(t, x0.ndim) / t_normalize
        a_t, b_t, c_t = self.get_abc(t)
        samples = a_t * xT + b_t * x0 + c_t * noise

        return samples

    def sample_step(self, t, t_prev, xt, x0, xT, t_normalize = 1000.0):
        # x_{t-1} ~ q(x_{t-1}|x_0, x_t, x_T)
        # x_0 should be the output of the denoiser D(x_t, t, x_T)

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
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: dict):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()
        print("Using DDBM as Trajectory Head!")

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.loss_computer = DDBMLossComputer(config=config)

        # 8 waypoint on navsim
        self.x_mean = [2.24042953e+00,4.46415814e+00,6.66798219e+00,8.84676197e+00,1.09943807e+01,1.31043913e+01,1.51712179e+01,1.71905959e+01]
        self.y_mean = [1.41745489e-02,5.99479783e-02,1.36759257e-01,2.42784730e-01,3.75173420e-01,5.30464399e-01,7.05211177e-01,8.95890308e-01]

        self.x_std = [1.53200873,3.01096188,4.43669179,5.81206158,7.14803658,8.46113169,9.76858244,11.08511972]
        self.y_std = [0.06845407,0.27803566,0.62667644,1.10570293,1.70445957,2.41085513,3.21227351,4.09599286]

        self.mean = torch.tensor([self.x_mean, self.y_mean], dtype=torch.float32).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        self.std = torch.tensor([self.x_std, self.y_std], dtype=torch.float32).transpose(0, 1).unsqueeze(0).unsqueeze(0)

        self.diffusion_scheduler = DDBMScheduler(
            beta_d=config.beta_d,
            beta_min=0.1,
            T=1.0,
        )

        plan_anchor = np.load(config.plan_anchor_path)
        print(config.plan_anchor_path, plan_anchor.shape)
        #plan_anchor[:,:-1,0] += config.ego_extent_x
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 64*(num_poses)), 
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )

        diff_decoder_layer_cls = CustomTransformerDecoderLayerCls(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )

        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.diff_decoder_cls = CustomTransformerDecoderCls(diff_decoder_layer_cls, 2)

        self.scalar_norm = config.use_scalar_norm

        print(f"normalization method: using scalar norm is {self.scalar_norm}!")
        print(f"beta_d: {config.beta_d}")

        # self.loss_computer = LossComputer(config)

    def norm_odo_simple(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    

    def denorm_odo_simple(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    

    def norm_odo(self, odo_info_fut):
        if self.scalar_norm:
            return self.norm_odo_simple(odo_info_fut)
        else:
            return (odo_info_fut - self.mean.to(odo_info_fut.device)) / self.std.to(odo_info_fut.device)

    def denorm_odo(self, odo_info_fut):
        if self.scalar_norm:
            return self.denorm_odo_simple(odo_info_fut)
        else:
            return odo_info_fut*self.std.to(odo_info_fut.device)+self.mean.to(odo_info_fut.device)

    def forward(self,ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,targets=None,global_img=None,mask=None,y_0=None):
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape, targets=targets,global_img=global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,global_img)

    def forward_train(self, ego_query, agents_query, bev_feature,bev_spatial_shape, targets=None, global_img=None):
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)

        x_0 = targets["trajectory"]
        x_0 = x_0.unsqueeze(1)[...,:2].repeat(1, plan_anchor.shape[1], 1, 1)

        odo_info_fut = self.norm_odo(x_0)
        odo_plan_anchor = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            1, 1001,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)

        noisy_traj_points = self.diffusion_scheduler.add_noise(
            x0=odo_info_fut,
            xT=odo_plan_anchor,
            noise=noise,
            t=timesteps,
        ).float()
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]; ak_ego_fut_mode = plan_anchor.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)

        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor,hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs,ak_ego_fut_mode,-1)

        # begin the stacked decoder
        poses_reg_list, _ = self.diff_decoder(traj_feature, noisy_traj_points, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img)
        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3) # align back to num_pose, 3 to obtain same shape as traj

        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)

        # compute and retuen loss here
        ret_traj_loss = 0
        reg_traj_loss = 0
        cls_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            traj_cls_loss, traj_reg_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor, best_reg)
            trajectory_loss = traj_cls_loss + traj_reg_loss

            cls_traj_loss += traj_cls_loss
            reg_traj_loss += traj_reg_loss
            ret_traj_loss += trajectory_loss

        return {"poses_reg_list": poses_reg_list, "poses_cls_list": poses_cls_list,
                "trajectory": best_reg, "trajectory_loss": ret_traj_loss,
                "trajectory_loss_dict":{
                    "cls_traj_loss": cls_traj_loss,
                    "reg_traj_loss": reg_traj_loss,
                }}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,global_img):
        device = ego_query.device
        bs = ego_query.shape[0]
        
        step_num = 20
        ts = torch.arange(0, 1001, 1000//step_num).to(device)
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)

        ak_ego_fut_mode = plan_anchor.shape[1]
        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor,hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs,ak_ego_fut_mode,-1)

        poses_cls_list = self.diff_decoder_cls(ak_traj_feature, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, global_img)
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        # align back to num_pose, 3 to obtain same shape as traj
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3) 

        xt_prev = plan_anchor

        for i in range(step_num, 0, -1):

            xt = xt_prev

            t = ts[i] * torch.ones([bs], device=device)
            t_prev = ts[i-1] * torch.ones([bs], device=device)

            traj_pos_embed = gen_sineembed_for_position(xt,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,xt.shape[1],-1)
            # embed the timesteps
            time_embed = self.time_mlp(t)
            time_embed = time_embed.view(bs,1,-1)

            poses_reg_list, _ = self.diff_decoder(traj_feature, xt, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img)

            x0_mean = poses_reg_list[-1][...,:2] # slice for diffusion process in waypoint
            head0_mean = poses_reg_list[-1][...,[-1]]

            xt_prev_norm = self.diffusion_scheduler.sample_step(
                t=t,
                t_prev=t_prev,
                xt=self.norm_odo(xt.clone()),
                x0=self.norm_odo(x0_mean),
                xT=self.norm_odo(plan_anchor.clone()),
            )

            xt_prev = self.denorm_odo(xt_prev_norm)

        xt_prev_ = torch.cat([xt_prev, head0_mean], dim=-1)
        x0_sample = torch.gather(xt_prev_, 1, mode_idx).squeeze(1)

        return {"trajectory": x0_sample} # align with navsim format


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
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
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls

class DiffMotionPlanningRefinementModuleCls(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModuleCls, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)

        return plan_cls

class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)

        self.dropout_T = nn.Dropout(0.1)
        self.dropout1_T = nn.Dropout(0.1)

        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses, 
            config=config,
            in_bev_dims=256,
        )
        self.cross_bev_attention_T = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses, 
            config=config,
            in_bev_dims=256,
        )

        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_agent_attention_T = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        ) # add agent attention so the anchor branch

        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.cross_ego_attention_T = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn_T = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)

        self.norm1_T = nn.LayerNorm(config.tf_d_model)
        self.norm2_T = nn.LayerNorm(config.tf_d_model)
        self.norm3_T = nn.LayerNorm(config.tf_d_model)

        self.time_modulation = ModulationLayer(config.tf_d_model*2,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model*2,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    # Shu update decoder input
    def forward(self, 
                traj_feature, # no change
                noisy_traj_points,  # no change
                plan_anchor,
                bev_feature,  
                bev_spatial_shape, # no change
                agents_query,
                ego_query, 
                time_embed=None, 
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature_T = self.cross_bev_attention_T(traj_feature,plan_anchor,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]) # traj_feature=[bs,20,256]; target_point_emb=[bs,1,256]
        traj_feature_T = traj_feature_T + self.dropout(self.cross_agent_attention_T(traj_feature_T, agents_query, agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        traj_feature_T = self.norm1_T(traj_feature_T)

        # cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        traj_feature_T = traj_feature_T + self.dropout1_T(self.cross_ego_attention_T(traj_feature_T, ego_query, ego_query)[0])
        traj_feature_T = self.norm2_T(traj_feature_T)

        # feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature_T = self.norm3_T(self.ffn_T(traj_feature_T))
        traj_feature = torch.cat([traj_feature, traj_feature_T], dim=-1)
        # modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None,global_img=global_img)
        
        # predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls 


class CustomTransformerDecoderLayerCls(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
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
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder_cls = DiffMotionPlanningRefinementModuleCls(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses+1, # Shu add placeholder for speed prediction
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points,  
                bev_feature,  
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed=None, 
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        # agent_query is now target_point
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]) 
        traj_feature = self.norm1(traj_feature)
        
        # cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))

        poses_cls = self.task_decoder_cls(traj_feature)

        return poses_cls

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points,
                plan_anchor,
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                #status_encoding, # not used, remove 
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, plan_anchor, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class CustomTransformerDecoderCls(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                #status_encoding, # not used, remove 
                global_img=None):
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, global_img)
            poses_cls_list.append(poses_cls)
        return poses_cls_list

class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays."""

    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding boxes in logs."""

    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _HEADING = 6

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def HEIGHT(cls):
        return cls._HEIGHT

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)

    @classmethod
    @property
    def DIMENSION(cls):
        # assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls._LENGTH, cls._HEIGHT + 1)


class LidarIndex(IntEnum):
    """Intenum for lidar point cloud arrays."""

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def INTENSITY(cls):
        return cls._INTENSITY

    @classmethod
    @property
    def RING(cls):
        return cls._RING

    @classmethod
    @property
    def ID(cls):
        return cls._ID

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)
