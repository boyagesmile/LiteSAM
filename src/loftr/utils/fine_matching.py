import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger

# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module
import torch.nn.init as init

def exists(v):
    return v is not None


# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim, expansion_factor=1.):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, dim, bias=False) if expansion_factor != 1. else Identity()

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden


# 使用循环 GRU 的偏移计算模块
class HeatmapMinGRUOffset(nn.Module):
    def __init__(self, dim=9, hidden_size=16, n_iters=3):
        super().__init__()
        self.feature_extractors = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_iters)
        ])
        self.grus = nn.ModuleList([
            minGRU(dim, expansion_factor=hidden_size / dim) for _ in range(n_iters)
        ])
        self.fc1s = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(n_iters)
        ])
        self.fc2 = nn.Linear(hidden_size, 2)  # 输出 (x, y) 偏移量
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.n_iters = n_iters  # GRU 的循环次数
        # self._initialize_weights()

    def _initialize_weights(self):
        """显式初始化模型中的权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 针对 fc1 层，使用 He 初始化
                if module in self.fc1s:
                    for fc1 in self.fc1s:
                        init.kaiming_uniform_(fc1.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                        if fc1.bias is not None:
                            init.zeros_(fc1.bias)

                # 针对 fc2 层，使用 Xavier 初始化
                elif module == self.fc2:
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.zeros_(module.bias)

                # 针对其他 Linear 层，使用 Xavier 初始化
                else:
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.zeros_(module.bias)
            elif isinstance(module, minGRU):
                # 初始化 GRU 模块的权重
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        init.kaiming_uniform_(param, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                    elif 'bias' in name:
                        init.zeros_(param)

    def forward(self, heatmap):
        B, _ = heatmap.size()  # (B, 3, 3)

        # 将 3x3 热图展平成 9 个元素
        heatmap_seq = heatmap.view(B, 1, 9)  # 每个样本一个时间步，每步9个特征

        # 初始化隐状态
        prev_hidden = None

        # 多次循环 GRU 计算
        for i in range(self.n_iters):
            heatmap_seq = self.feature_extractors[i](heatmap_seq)
            _, prev_hidden = self.grus[i](heatmap_seq, prev_hidden, return_next_prev_hidden=True)
            prev_hidden = self.fc1s[i](prev_hidden)

        # 将最后的隐状态映射为 (x, y) 偏移量
        offsets = self.relu(prev_hidden)  # (B, 2)
        offsets = torch.tanh(self.fc2(offsets.squeeze(1)))

        return offsets


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        # d_model = config['backbone']['block_dims'][0] -self.local_regress_slicedim
        self.fp16 = config['half']
        self.validate = False
        self.update_block = HeatmapMinGRUOffset(dim=9, hidden_size=16, n_iters=4)

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            data.update({
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            feat_f0, feat_f1 = feat_0[..., :-self.local_regress_slicedim], feat_1[..., :-self.local_regress_slicedim]
            feat_ff0, feat_ff1 = feat_0[..., -self.local_regress_slicedim:], feat_1[..., -self.local_regress_slicedim:]
            feat_f0, feat_f1 = feat_f0 / C ** .5, feat_f1 / C ** .5
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1)
            conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim) ** .5)

        # MNN
        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W + 2, self.W + 2)
        softmax_matrix_f = softmax_matrix_f[..., 1:-1, 1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'conf_matrix_f': softmax_matrix_f})

        # compute pixel-level absolute kpt coords
        self.get_fine_ds_match(softmax_matrix_f, data)

        if data['idx_l'].numel() == 0 or ('m_ids_f' not in data and (self.training or self.validate)):
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return
        # sub-pixel
        if self.training:
            m_ids, idx_l, idx_r_iids, idx_r_jids = data['m_ids_f'], data['i_ids_f'], data['j_ids_f_di'], data[
                'j_ids_f_dj']
        else:
            idx_l, idx_r = data['idx_l'], data['idx_r']
            m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
            m_ids = m_ids[:len(data['mconf'])]
            idx_r_iids, idx_r_jids = idx_r // W, idx_r % W  # 图像宽高的位置
            m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(
                -1), idx_r_jids.reshape(-1)

        # 局部网格 3*3的网格
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long)  # [1, 3, 3, 2]

        m_ids = m_ids[..., None, None].expand(-1, 3, 3)
        idx_l = idx_l[..., None, None].expand(-1, 3, 3)  # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[..., None, None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return
        # compute second-stage heatmap
        # 取前后一个点的置信度， 3*3矩阵
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W + 2, self.W + 2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)  # M, ww, 9
        xy = self.update_block(conf_matrix_ff)
        #  self.validate
        if self.training:
            data.update({'expec_f': xy})
            data.update({
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:, None, :].expand(-1, -1,
                                                                                                        2).reshape(-1,
                                                                                                                   2) if 'scale0' in data else scale

        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(xy, data, scale1)

    def get_fine_match_local(self, coords_normed, data, scale1):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)
        # 1.0?
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']), ...]  # 展平 m
        val, idx = torch.max(conf_matrix, dim=-1)  # 取dim=-1维度的最大值
        idx = idx[:, None]  # 补一个维度
        idx_l, idx_r = idx // WW, idx % WW  # 行、列坐标对应两个图的位置

        data.update({'idx_l': idx_l, 'idx_r': idx_r})
        # 生成坐标网格，大小为（w,w,2），并将坐标中心平移到网格中心
        # 为啥加0.5？
        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device,
                                   dtype=torch.float16) - W // 2 + 0.5  # kornia >= 0.5.1
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        # 偏移量, expend2? 偏移是0~（w-1）
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale

        if torch.is_tensor(scale0) and scale0.numel() > 1:  # scale0 is a tensor
            mkpts0_f = (data['mkpts0_c'][:, None, :] + (
                    delta_l * scale0[:len(data['mconf']), ...][:, None, :])).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + (
                    delta_r * scale1[:len(data['mconf']), ...][:, None, :])).reshape(-1, 2)
        else:  # scale0 is a float
            mkpts0_f = (data['mkpts0_c'][:, None, :] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:, None, :] + (delta_r * scale1)).reshape(-1, 2)

        data.update({
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f
        })
