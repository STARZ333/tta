from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
from collections import deque
import logging
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
from torchvision.transforms import InterpolationMode
import random
import numpy as np

logger = logging.getLogger(__name__)


class StatisticsMemoryManager:
    def __init__(self, max_memory_size=10, domain_shift_threshold=2.0):
        """
        用于管理BN统计记忆队列并检测域切换。
        """
        self.max_memory_size = max_memory_size
        self.domain_shift_threshold = domain_shift_threshold

        self.mean_queue = deque(maxlen=self.max_memory_size)
        self.var_queue = deque(maxlen=self.max_memory_size)

    def update_memory(self, current_mean, current_var):
        self.mean_queue.append(current_mean.cpu())
        self.var_queue.append(current_var.cpu())

    def get_memory_statistics(self):
        if len(self.mean_queue) == 0:
            return None, None
        mean_stack = torch.stack(list(self.mean_queue), dim=0)  # [M, C]
        mu_mem = mean_stack.mean(dim=0)  # [C]

        var_stack = torch.stack(list(self.var_queue), dim=0)    # [M, C]
        var_mean = var_stack.mean(dim=0)
        mean_sq = (mean_stack ** 2).mean(dim=0)

        var_mem = var_mean + mean_sq - (mu_mem ** 2)
        return mu_mem, var_mem

    def compute_z_scores(self, current_mean, current_var):
        """
        计算 z_mu, z_var 的辅助函数。
        返回: (z_mu, z_var) 或 (None, None)
        """
        if len(self.mean_queue) < 2:
            return None, None  # 队列数据太少，不足以计算

        mu_mem, var_mem = self.get_memory_statistics()
        if mu_mem is None or var_mem is None:
            return None, None

        # 与记忆统计对比
        dist_mu_in = torch.norm(current_mean - mu_mem, p=2) ** 2
        dist_var_in = torch.norm(current_var - var_mem, p=2) ** 2

        dist_mu_list = []
        dist_var_list = []
        for (m, v) in zip(self.mean_queue, self.var_queue):
            d_mu = torch.norm(m - mu_mem, p=2) ** 2
            d_var = torch.norm(v - var_mem, p=2) ** 2
            dist_mu_list.append(d_mu.item())
            dist_var_list.append(d_var.item())

        dist_mu_list = torch.tensor(dist_mu_list)
        dist_var_list = torch.tensor(dist_var_list)

        mean_dist_mu = dist_mu_list.mean()
        std_dist_mu = dist_mu_list.std(unbiased=False)
        mean_dist_var = dist_var_list.mean()
        std_dist_var = dist_var_list.std(unbiased=False)

        z_mu, z_var = 0.0, 0.0
        if std_dist_mu > 1e-6:
            z_mu = (dist_mu_in - mean_dist_mu) / (std_dist_mu + 1e-5)
        if std_dist_var > 1e-6:
            z_var = (dist_var_in - mean_dist_var) / (std_dist_var + 1e-5)

        return float(z_mu), float(z_var)

    def detect_domain_shift(self, current_mean, current_var):
        """
        返回 (is_shift, z_mu, z_var):
          - is_shift: bool, 是否判定域切换
          - z_mu, z_var: 对应的z分数(可用于分析或可视化)
        """
        z_mu, z_var = self.compute_z_scores(current_mean, current_var)
        if z_mu is None or z_var is None:
            return False, None, None

        # 如果 abs(z_mu) or abs(z_var) 超过阈值就视为域切换
        is_shift = (abs(z_mu) > self.domain_shift_threshold) or (abs(z_var) > self.domain_shift_threshold)

        return is_shift, z_mu, z_var

    def reset_memory(self):
        self.mean_queue.clear()
        self.var_queue.clear()


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # === 新增: 每次 forward() 都+1，用于标识 micro-batch ===
        self.global_forward_count = 0  

        # 记录检测到的域切换时刻
        self.detected_shifts = []

        # 存储 z_mu, z_var 以及对应的 micro-batch id
        self.z_mu_list = []
        self.z_var_list = []
        self.z_batch_list = []

        # 复制初始模型和优化器状态
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.transform = get_tta_transforms()
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

        # BN统计管理器
        self.stats_manager = StatisticsMemoryManager(
            max_memory_size=10,
            domain_shift_threshold=2.0
        )
        self.register_bn_hooks()

    def register_bn_hooks(self):
        target_bn = None
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                target_bn = m
                break
        if target_bn is not None:
            target_bn.register_forward_hook(self.bn_forward_hook)
        else:
            logger.warning("No BN layer found to track BN stats.")

    def bn_forward_hook(self, module, input, output):
        x = input[0]
        dim = [0] + list(range(2, x.dim()))
        current_mean = x.mean(dim=dim)
        current_var = x.var(dim=dim, unbiased=False)
        self.stats_manager.update_memory(current_mean.detach(), current_var.detach())

    def forward(self, x):
        """
        每次 forward() 都视作一个 micro-batch. 
        """
        # 记录 micro-batch编号
        self.global_forward_count += 1
        logger.info(f"[CoTTA] Processing micro-batch #{self.global_forward_count}.")

        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs_ema = self.forward_and_adapt(x)

        # 每次 forward 完成后，做域检测（对 micro-batch）
        self._detect_domain_shift_if_needed()

        return outputs_ema

    def _detect_domain_shift_if_needed(self):
        # 需要至少1条统计才可以做检测
        if len(self.stats_manager.mean_queue) < 1:
            return

        current_mean = self.stats_manager.mean_queue[-1]
        current_var = self.stats_manager.var_queue[-1]

        is_shift, z_mu, z_var = self.stats_manager.detect_domain_shift(current_mean, current_var)

        # 记录 z_mu, z_var 到列表
        if z_mu is not None and z_var is not None and self.global_forward_count % 10 == 0:
            self.z_mu_list.append(z_mu)
            self.z_var_list.append(z_var)
            # 这里用 global_forward_count 做 x 轴
            self.z_batch_list.append(self.global_forward_count)

        if is_shift:
            logger.info(f"[CoTTA] Domain shift detected at micro-batch #{self.global_forward_count}. "
                        "Resetting stats.")
            self.reset()
            self.stats_manager.reset_memory()
            self.detected_shifts.append(self.global_forward_count)

    def reset(self):
        """
        清空统计记忆。若想恢复模型到初始参数，可取消注释 load_model_and_optimizer。
        """
        self.stats_manager.reset_memory()
        # load_model_and_optimizer(self.model, self.optimizer,
        #                          self.model_state, self.optimizer_state)
        # self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)

        N = 8
        outputs_emas = []
        for _ in range(N):
            outputs_aug = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(outputs_aug)

        if anchor_prob.mean(0) < self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model_ema = update_ema_variables(self.model_ema, self.model, self.mt)

        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    if f"{nm}.{npp}" in self.model_state:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
                print(nm, np)
    return params, names

def collect_bn_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(f"Selected parameter: {nm}.{np}")
    return params, names

def collect_specific_params(model, update_layers):
    params = []
    names = set()
    print(f"Update layers: {update_layers}")
    print("All module names in the model:")
    for nm, m in model.named_modules():
        print(nm)
    for nm, m in model.named_modules():
        if any(nm == f"stage_{layer_num}" or nm.startswith(f"stage_{layer_num}.") for layer_num in update_layers):
            for np, p in m.named_parameters():
                param_name = f"{nm}.{np}"
                if param_name not in names and p.requires_grad:
                    params.append(p)
                    names.add(param_name)
                    print(f"Selected parameter: {param_name}")
    return params, list(names)

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model

def check_model(model):
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update."
    assert not has_all_params, "tent should not update all params."
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
