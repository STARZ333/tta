import torch
import math
from copy import deepcopy
from torch import nn

# -------------------- 公共工具 -------------------- #
def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
    group_avgs = []
    for i, lbl_idcs in enumerate(lbl_group):
        group_avgs.append(feats[lbl_idcs].mean(axis=0).unsqueeze(0))
    return lbl_uniq, group_avgs


class DivergenceScore(nn.Module):
    """原 PeTTA 中的损失，未改动"""
    def __init__(self, src_prototype, src_prototype_cov):
        super().__init__()
        self.src_proto = src_prototype
        self.src_proto_cov = src_prototype_cov

        def GSSLoss(input, target, target_cov):
            return ((input - target).pow(2) / (target_cov + 1e-6)).mean()

        self.lss = GSSLoss

    def forward(self, feats, pseudo_lbls):
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        return self.lss(
            torch.cat(group_avgs, dim=0),
            self.src_proto[lbl_uniq],
            self.src_proto_cov[lbl_uniq],
        )


class PrototypeMemory:
    """保持不变，用于原型平滑"""
    def __init__(self, src_prototype, num_classes):
        self.src_proto = src_prototype.squeeze(1)
        self.mem_proto = deepcopy(self.src_proto)
        self.num_classes = num_classes
        self.src_proto_l2 = torch.cdist(self.src_proto, self.src_proto, p=2)

    def update(self, feats, pseudo_lbls, nu=0.05):
        lbl_uniq = torch.unique(pseudo_lbls)
        lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl = lbl_uniq[i]
            batch_avg = feats[lbl_idcs].mean(axis=0)
            self.mem_proto[psd_lbl] = (1 - nu) * self.mem_proto[psd_lbl] + nu * batch_avg

    def get_mem_prototype(self):
        return self.mem_proto


# -------------------- MemoryItem -------------------- #
class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, true_label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.true_label = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def empty(self):
        return self.data == "empty"


# -------------------- PeTTA Memory (改良版) -------------------- #
class PeTTAMemory:
    def __init__(self,
                 capacity: int,
                 num_class: int,
                 lambda_t: float = 1.0,
                 lambda_u: float = 1.0,
                 lambda_d: float = 1.0):
        """
        capacity : 整体样本上限
        num_class: 类别数
        λ_t, λ_u, λ_d: 分别对应 age / uncertainty / distance 三个启发式权重
        """
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class

        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.lambda_d = lambda_d

        # data[c] 存放类别 c 的 MemoryItem 列表
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    # ---------- 描述符计算（与 MyTTA 保持一致） ---------- #
    @staticmethod
    def _instance_descriptor(x: torch.Tensor):
        """(C,H,W) → (mean, var) 各为 (C,)"""
        mean = torch.mean(x, dim=(1, 2))
        var = torch.var(x, dim=(1, 2))
        return mean, var

    @staticmethod
    def _class_descriptor(class_items: list):
        """将类内所有样本拼成批量后算 (mean, var)"""
        if len(class_items) == 0:
            return None, None
        stack = torch.stack([itm.data for itm in class_items])          # (N,C,H,W)
        mean = torch.mean(stack, dim=(0, 2, 3))                         # (C,)
        var = torch.var(stack, dim=(0, 2, 3))                           # (C,)
        return mean, var

    @staticmethod
    def _descriptor_distance(inst_desc, class_desc):
        """拼接 mean ∥ var → 2C 向量做 L2 距离"""
        if class_desc[0] is None:
            return 0.0
        inst_vec = torch.cat(inst_desc)                                 # (2C,)
        class_vec = torch.cat(class_desc)                               # (2C,)
        return torch.norm(inst_vec - class_vec, p=2).item()

    # ---------- 基本统计 ---------- #
    def get_occupancy(self) -> int:
        return sum(len(lst) for lst in self.data)

    def per_class_dist(self):
        return [len(lst) for lst in self.data]

    def get_majority_classes(self):
        per_cls = self.per_class_dist()
        max_occ = max(per_cls)
        return [i for i, occ in enumerate(per_cls) if occ == max_occ]

    # ---------- 启发式得分 ---------- #
    def heuristic_score(self, age, uncertainty, distance):
        term_t = self.lambda_t * (1 / (1 + math.exp(-age / self.capacity)))
        term_u = self.lambda_u * (uncertainty / math.log(self.num_class))
        term_d = self.lambda_d * distance
        return term_t + term_u + term_d

    # ---------- 核心流程 ---------- #
    def add_instance(self, instance):
        """
        instance = (x, prediction, uncertainty, true_label)
        先用描述符距离打分，如通过删除逻辑后才真正插入。
        """
        assert len(instance) == 4
        x, pred_cls, uncertainty, true_label = instance

        inst_desc = self._instance_descriptor(x)
        class_desc = self._class_descriptor(self.data[pred_cls])
        distance = self._descriptor_distance(inst_desc, class_desc)

        new_item = MemoryItem(x, uncertainty, age=0, true_label=true_label)
        new_score = self.heuristic_score(age=0,
                                         uncertainty=uncertainty,
                                         distance=distance)

        if self._remove_to_fit(pred_cls, new_score):
            self.data[pred_cls].append(new_item)

        self._increase_all_age()

    # ---------- 删除策略 ---------- #
    def _remove_to_fit(self, cls: int, new_score: float) -> bool:
        """
        若类别 cls 未饱和且总容量未满：直接存。  
        否则按启发式分数从待删集合里挑一个分数最高的样本删除。
        """
        class_list = self.data[cls]
        class_occ = len(class_list)
        total_occ = self.get_occupancy()

        if class_occ < self.per_class:
            if total_occ < self.capacity:
                return True
            else:
                # 总容量满：从“多数类”里删
                majority = self.get_majority_classes()
                return self._remove_from_classes(majority, new_score)
        else:
            # 该类已满：只能在本类删
            return self._remove_from_classes([cls], new_score)

    def _remove_from_classes(self, classes: list, score_base: float) -> bool:
        """
        在 classes 中找启发式分数最高的旧样本，若 > score_base 则删。
        """
        cand_class, cand_idx, cand_score = None, None, None

        for c in classes:
            c_desc = self._class_descriptor(self.data[c])
            for idx, itm in enumerate(self.data[c]):
                dist = self._descriptor_distance(
                    self._instance_descriptor(itm.data), c_desc)
                score = self.heuristic_score(itm.age, itm.uncertainty, dist)
                if cand_score is None or score > cand_score:
                    cand_class, cand_idx, cand_score = c, idx, score

        # 若最佳候选分数更高 → 删除它
        if cand_class is not None and cand_score > score_base:
            dead = self.data[cand_class].pop(cand_idx)
            del dead
            return True
        return False if cand_class is not None else True

    # ---------- 工具 ---------- #
    def _increase_all_age(self):
        for class_list in self.data:
            for itm in class_list:
                itm.increase_age()

    def get_memory(self):
        all_data, all_age = [], []
        for class_list in self.data:
            for itm in class_list:
                all_data.append(itm.data)
                all_age.append(itm.age / self.capacity)    # 归一化
        return all_data, all_age
