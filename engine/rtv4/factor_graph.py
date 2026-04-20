"""
Lightweight factor-graph utilities for RT-DETRv4.

This module provides:
1) Post-hoc NMS-free refinement for inference.
2) Differentiable factor-graph regularization loss for training.
"""

import torch
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_iou


def _safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_stats_cxcywh(boxes_cxcywh: torch.Tensor):
    w = boxes_cxcywh[..., 2].clamp(min=1e-6)
    h = boxes_cxcywh[..., 3].clamp(min=1e-6)
    log_area = torch.log((w * h).clamp(min=1e-12))
    log_ar = torch.log((w / h).clamp(min=1e-6))
    centers = boxes_cxcywh[..., :2]
    return log_area, log_ar, centers


def _build_knn_weights(centers: torch.Tensor,
                       labels: torch.Tensor,
                       num_neighbors: int,
                       center_sigma: float) -> torch.Tensor:
    n = centers.shape[0]
    if n <= 1:
        return centers.new_zeros((n, n))

    diff = centers[:, None, :] - centers[None, :, :]
    dist2 = (diff * diff).sum(dim=-1)
    dist2.fill_diagonal_(1e9)

    k = max(1, min(int(num_neighbors), n - 1))
    knn_vals, knn_idx = torch.topk(dist2, k=k, largest=False, dim=-1)

    denom = max(float(center_sigma) * float(center_sigma), 1e-6)
    knn_w = torch.exp(-0.5 * knn_vals / denom)

    label_eq = (labels[:, None] == labels[None, :]).to(knn_w.dtype)
    knn_w = knn_w * label_eq.gather(dim=1, index=knn_idx)

    weights = centers.new_zeros((n, n))
    weights.scatter_(dim=1, index=knn_idx, src=knn_w)
    weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return weights / weights_sum


class FactorGraphPostRefiner:
    """Lightweight NMS-free posterior refiner.

    This refiner updates candidate scores and boxes through a compact factor graph
    with prior, motion, and observation terms.
    """

    def __init__(self,
                 enabled: bool = False,
                 num_neighbors: int = 6,
                 num_iters: int = 2,
                 prior_weight: float = 0.20,
                 motion_weight: float = 0.35,
                 obs_weight: float = 0.45,
                 box_refine_weight: float = 0.12,
                 iou_gate: float = 0.05,
                 center_sigma: float = 0.18,
                 prior_log_area_mean: float = -6.5,
                 prior_log_area_std: float = 1.3,
                 prior_log_ar_mean: float = 0.0,
                 prior_log_ar_std: float = 1.0):
        self.enabled = bool(enabled)
        self.num_neighbors = int(num_neighbors)
        self.num_iters = int(num_iters)
        self.prior_weight = float(prior_weight)
        self.motion_weight = float(motion_weight)
        self.obs_weight = float(obs_weight)
        self.box_refine_weight = float(box_refine_weight)
        self.iou_gate = float(iou_gate)
        self.center_sigma = float(center_sigma)

        self.prior_log_area_mean = float(prior_log_area_mean)
        self.prior_log_area_std = max(float(prior_log_area_std), 1e-6)
        self.prior_log_ar_mean = float(prior_log_ar_mean)
        self.prior_log_ar_std = max(float(prior_log_ar_std), 1e-6)

    @torch.no_grad()
    def refine(self,
               scores: torch.Tensor,
               labels: torch.Tensor,
               boxes_xyxy_abs: torch.Tensor,
               orig_target_sizes: torch.Tensor):
        if (not self.enabled) or scores.numel() == 0:
            return scores, boxes_xyxy_abs

        bsz, num_boxes = scores.shape[:2]
        if num_boxes <= 1:
            return scores, boxes_xyxy_abs

        whwh = orig_target_sizes.repeat(1, 2).unsqueeze(1).clamp(min=1e-6)
        boxes_norm = boxes_xyxy_abs / whwh
        boxes_cxcywh = _xyxy_to_cxcywh(boxes_norm)

        refined_scores = []
        refined_boxes = []

        for b in range(bsz):
            s = scores[b]
            lab = labels[b].long()
            box_xyxy = boxes_norm[b]
            box_cxcywh = boxes_cxcywh[b]

            log_area, log_ar, centers = _box_stats_cxcywh(box_cxcywh)
            prior_term = -0.5 * ((log_area - self.prior_log_area_mean) / self.prior_log_area_std) ** 2
            prior_term = prior_term - 0.5 * ((log_ar - self.prior_log_ar_mean) / self.prior_log_ar_std) ** 2

            w_knn = _build_knn_weights(centers, lab, self.num_neighbors, self.center_sigma)
            z_obs = _safe_logit(s)
            z = z_obs.clone()

            for _ in range(max(1, self.num_iters)):
                z_motion = torch.matmul(w_knn, z)
                denom = self.obs_weight + self.prior_weight + self.motion_weight + 1e-8
                z = (self.obs_weight * z_obs + self.prior_weight * prior_term + self.motion_weight * z_motion) / denom

            s_ref = torch.sigmoid(z)

            if self.box_refine_weight > 0:
                iou_mat, _ = box_iou(box_xyxy, box_xyxy)
                iou_mat.fill_diagonal_(0)
                iou_mask = (iou_mat > self.iou_gate).to(box_xyxy.dtype)
                w_geom = w_knn * iou_mask
                w_geom_sum = w_geom.sum(dim=1, keepdim=True).clamp(min=1e-8)
                w_geom = w_geom / w_geom_sum
                neigh_box = torch.matmul(w_geom, box_cxcywh)
                gate = (s_ref - s_ref.mean()).sigmoid().unsqueeze(-1)
                blend = (self.box_refine_weight * gate).clamp(min=0.0, max=0.5)
                box_cxcywh = (1.0 - blend) * box_cxcywh + blend * neigh_box
                box_xyxy = box_cxcywh_to_xyxy(box_cxcywh).clamp(min=0.0, max=1.0)

            refined_scores.append(s_ref)
            refined_boxes.append(box_xyxy)

        scores_out = torch.stack(refined_scores, dim=0)
        boxes_out = torch.stack(refined_boxes, dim=0) * whwh
        return scores_out, boxes_out


def factor_graph_training_loss(outputs,
                               targets,
                               indices,
                               num_boxes,
                               params=None):
    """Differentiable factor-graph regularization loss.

    Factors:
    - prior factor on matched box area/aspect distribution;
    - motion factor on local consistency among matched predictions;
    - observation factor from cls confidence and IoU quality.
    """
    if params is None:
        params = {}

    w_prior = float(params.get('prior_weight', 0.20))
    w_motion = float(params.get('motion_weight', 0.35))
    w_obs = float(params.get('obs_weight', 0.45))
    num_neighbors = int(params.get('num_neighbors', 6))
    center_sigma = float(params.get('center_sigma', 0.18))

    if indices is None:
        device = outputs['pred_boxes'].device
        return {'loss_fgo': torch.tensor(0.0, device=device)}

    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    if src_idx.numel() == 0:
        device = outputs['pred_boxes'].device
        return {'loss_fgo': torch.tensor(0.0, device=device)}

    src_boxes = outputs['pred_boxes'][batch_idx, src_idx]
    src_logits = outputs['pred_logits'][batch_idx, src_idx]
    tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
    tgt_labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)], dim=0).long()

    if src_logits.shape[-1] == 1:
        cls_logit = src_logits.squeeze(-1)
    else:
        cls_logit = src_logits.gather(dim=-1, index=tgt_labels.unsqueeze(-1)).squeeze(-1)

    ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
    iou_diag = torch.diag(ious).detach().clamp(min=0.0, max=1.0)
    obs_loss = F.binary_cross_entropy_with_logits(cls_logit, iou_diag, reduction='mean')

    pred_log_area, pred_log_ar, _ = _box_stats_cxcywh(src_boxes)
    tgt_log_area, tgt_log_ar, _ = _box_stats_cxcywh(tgt_boxes)
    area_mu = tgt_log_area.detach().mean()
    area_std = tgt_log_area.detach().std(unbiased=False).clamp(min=1e-4)
    ar_mu = tgt_log_ar.detach().mean()
    ar_std = tgt_log_ar.detach().std(unbiased=False).clamp(min=1e-4)
    prior_loss = (((pred_log_area - area_mu) / area_std) ** 2 + ((pred_log_ar - ar_mu) / ar_std) ** 2).mean()

    motion_loss_box = src_boxes.new_tensor(0.0)
    motion_loss_score = src_boxes.new_tensor(0.0)
    valid_groups = 0
    pred_score = torch.sigmoid(cls_logit)

    for b in range(len(targets)):
        m = (batch_idx == b)
        if m.sum() <= 1:
            continue
        b_boxes = src_boxes[m]
        b_scores = pred_score[m]
        b_labels = tgt_labels[m]

        _, _, centers = _box_stats_cxcywh(b_boxes)
        w_knn = _build_knn_weights(centers, b_labels, num_neighbors=num_neighbors, center_sigma=center_sigma)

        neigh_boxes = torch.matmul(w_knn, b_boxes)
        neigh_scores = torch.matmul(w_knn, b_scores.unsqueeze(-1)).squeeze(-1)
        motion_loss_box = motion_loss_box + F.smooth_l1_loss(b_boxes, neigh_boxes, reduction='mean')
        motion_loss_score = motion_loss_score + F.mse_loss(b_scores, neigh_scores, reduction='mean')
        valid_groups += 1

    if valid_groups > 0:
        motion_loss = motion_loss_box / valid_groups + 0.5 * (motion_loss_score / valid_groups)
    else:
        motion_loss = src_boxes.new_tensor(0.0)

    loss_fgo = w_prior * prior_loss + w_motion * motion_loss + w_obs * obs_loss

    return {'loss_fgo': loss_fgo}